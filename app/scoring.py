"""Deterministic risk scorers (no LLM)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from app.config import ReviewPriority
from app.utils import clip01, haversine_km, looks_like_biotag


_URGENCY = re.compile(
    r"\b(urgent|immediately|within\s+\d+\s*(hour|hr|minute)|verify\s+now|"
    r"account\s+(lock|locked|suspend|suspended)|action\s+required|"
    r"unusual\s+login|confirm\s+identity|customs\s+fee|prize|winner|"
    r"crypto|wallet|seed\s+phrase)\b",
    re.I,
)
_SHORTENER = re.compile(r"https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)/", re.I)
_SUSPICIOUS_TLDS = re.compile(r"\.(tk|ml|ga|cf|gq|zip|mov)\b", re.I)


def _domain_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        return host.split(":")[0]
    except Exception:
        return ""


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\"'<>]+", text or "", flags=re.I)


def _lookalike_domain_score(domain: str) -> float:
    if not domain:
        return 0.0
    d = domain.lower()
    score = 0.0
    if re.search(r"(paypa|amazon|apple|google|microsoft|netflix)[^.\s]*\d", d):
        score += 0.35
    if re.sub(r"[0-9l]", "", d) != re.sub(r"[0-9o]", "", d.replace("0", "o")):
        pass
    pairs = [("paypal", "paypa"), ("amazon", "amaz"), ("apple", "app1e"), ("google", "g00gle")]
    for legit, pat in pairs:
        if pat in d and legit not in d:
            score += 0.25
    if _SHORTENER.search(d):
        score += 0.15
    if _SUSPICIOUS_TLDS.search(d):
        score += 0.2
    return clip01(score)


def score_phishing_text(text: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    if not text:
        return 0.0, flags
    t = text.lower()
    s = 0.0
    if _URGENCY.search(t):
        s += 0.25
        flags.append("urgency_language")
    for url in _extract_urls(text):
        dom = _domain_from_url(url)
        ls = _lookalike_domain_score(dom)
        if ls > 0.2:
            flags.append(f"suspicious_domain:{dom[:40]}")
        s += ls * 0.35
    if "verify payment" in t or "confirm billing" in t:
        s += 0.1
        flags.append("billing_verify_phrase")
    return clip01(s), flags


def _user_email_guess(row: pd.Series) -> str:
    fn = str(row.get("first_name") or "").strip().lower()
    ln = str(row.get("last_name") or "").strip().lower()
    if fn and ln:
        return f"{fn}.{ln}"
    return fn or ln


def build_message_risk_tables(
    mails: pd.DataFrame,
    sms: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Per timestamp index of global phishing intensity (joined in window per tx)."""
    rows = []
    guesses = {_user_email_guess(r): idx for idx, r in users.iterrows()}
    if not mails.empty:
        for _, m in mails.iterrows():
            ts = m.get("mail_timestamp")
            if pd.isna(ts):
                continue
            body = f"{m.get('subject','')} {m.get('body','')}"
            score, flags = score_phishing_text(body)
            to_a = str(m.get("to_addr") or "").lower()
            targeted = any(g and g in to_a for g in guesses if g)
            rows.append({"ts": ts, "channel": "mail", "score": score, "flags": flags, "targeted": targeted})
    if not sms.empty:
        for _, s in sms.iterrows():
            ts = s.get("sms_timestamp")
            if pd.isna(ts):
                continue
            body = str(s.get("sms_text") or "")
            score, flags = score_phishing_text(body)
            rows.append({"ts": ts, "channel": "sms", "score": score, "flags": flags, "targeted": True})
    return pd.DataFrame(rows)


def _window_score(msg_df: pd.DataFrame, t: pd.Timestamp, hours: float = 72.0) -> tuple[float, list[str]]:
    if msg_df.empty or pd.isna(t):
        return 0.0, []
    lo = t - pd.Timedelta(hours=hours)
    hi = t + pd.Timedelta(hours=hours)
    sub = msg_df[(msg_df["ts"] >= lo) & (msg_df["ts"] <= hi)]
    if sub.empty:
        return 0.0, []
    mx = float(sub["score"].max())
    top = sub.loc[sub["score"].idxmax()]
    flags: list[str] = []
    for x in list(top.get("flags") or []):
        if x not in flags:
            flags.append(x)
    return mx, flags[:12]


def user_vulnerability_scores(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    scores = []
    for _, r in users.iterrows():
        desc = str(r.get("description") or "")
        s = 0.2
        dl = desc.lower()
        if "phish" in dl or "scam" in dl or "suspicious link" in dl:
            s += 0.25
        m = re.search(r"(\d{1,3})\s*%", desc)
        if m:
            try:
                pct = int(m.group(1))
                s += clip01(pct / 100.0) * 0.35
            except ValueError:
                pass
        if re.search(r"\b(one|once|1)\s+time\b.*year", dl) or re.search(r"once per year", dl):
            s += 0.05
        if re.search(r"\b(five|5)\s*time", dl) or re.search(r"fünfmal", dl):
            s += 0.05
        scores.append(clip01(s))
    out["vulnerability_score"] = scores
    return out


def _nearest_location_features(
    tx_row: pd.Series,
    locations: pd.DataFrame,
    res_lat: float,
    res_lng: float,
) -> tuple[float, float, float]:
    """Returns (dist_to_residence_km, min_ping_dist_km, travel_speed_kmh_est)."""
    if locations.empty or pd.isna(tx_row.get("timestamp")):
        return np.nan, np.nan, 0.0
    bio = tx_row.get("sender_biotag")
    sub = locations[locations["biotag"] == bio].copy() if pd.notna(bio) else locations.iloc[0:0]
    if sub.empty:
        return np.nan, np.nan, 0.0
    sub = sub.dropna(subset=["timestamp", "lat", "lng"])
    t0 = tx_row["timestamp"]
    win = sub[(sub["timestamp"] >= t0 - pd.Timedelta(hours=48)) & (sub["timestamp"] <= t0 + pd.Timedelta(hours=48))]
    if win.empty:
        win = sub
    dists = []
    for _, lr in win.iterrows():
        if np.isfinite(res_lat) and np.isfinite(res_lng):
            dists.append(haversine_km(res_lat, res_lng, float(lr["lat"]), float(lr["lng"])))
    dist_res = float(np.min(dists)) if dists else np.nan
    if len(win) >= 2:
        win = win.sort_values("timestamp")
        speeds = []
        a = win.iloc[:-1]
        b = win.iloc[1:]
        for i in range(len(win) - 1):
            r1, r2 = win.iloc[i], win.iloc[i + 1]
            dt_h = abs((r2["timestamp"] - r1["timestamp"]).total_seconds()) / 3600.0
            if dt_h < 1e-6:
                continue
            dkm = haversine_km(float(r1["lat"]), float(r1["lng"]), float(r2["lat"]), float(r2["lng"]))
            speeds.append(dkm / dt_h)
        speed_est = float(np.max(speeds)) if speeds else 0.0
    else:
        speed_est = 0.0
    ping_min = float(np.min([haversine_km(res_lat, res_lng, float(r["lat"]), float(r["lng"])) for _, r in win.iterrows()])) if len(win) else np.nan
    return dist_res, ping_min, speed_est


def score_transactions(tables: dict[str, Any]) -> pd.DataFrame:
    tx = tables["transactions"].copy()
    users = tables["users"].copy()
    locs = tables["locations"].copy()
    mails = tables["mails"].copy()
    sms = tables["sms"].copy()

    tx["sender_is_customer"] = tx["sender_iban"].isin(set(users["iban"]))
    bio_series = users.drop_duplicates("iban").set_index("iban")["biotag"]
    tx["sender_biotag"] = tx["sender_iban"].map(bio_series.to_dict())
    mask_bio = tx["sender_biotag"].isna() & tx["sender_id"].map(looks_like_biotag)
    tx.loc[mask_bio, "sender_biotag"] = tx.loc[mask_bio, "sender_id"]

    users_v = user_vulnerability_scores(users)
    vuln_map = users_v.set_index("iban")["vulnerability_score"].to_dict()
    sal_map = users_v.set_index("iban")["salary"].to_dict()
    lat_map = users_v.set_index("iban")["residence_lat"].to_dict()
    lng_map = users_v.set_index("iban")["residence_lng"].to_dict()
    city_map = users_v.set_index("iban")["residence_city"].astype(str).str.lower().to_dict()

    tx["user_vulnerability"] = tx["sender_iban"].map(lambda ib: vuln_map.get(ib, 0.35))
    tx["salary"] = tx["sender_iban"].map(lambda ib: float(sal_map.get(ib, 0.0) or 0.0))

    monthly_income = (tx["salary"] / 12.0).replace(0, np.nan)
    tx["amount_to_monthly_ratio"] = (tx["amount"] / (monthly_income + 1.0)).fillna(0.0)

    grp = tx.groupby("sender_iban", sort=False)["amount"]
    mean_amt = grp.transform("mean")
    std_amt = grp.transform("std").replace(0, np.nan)
    tx["amount_z"] = ((tx["amount"] - mean_amt) / (std_amt + 1e-6)).fillna(0.0)

    tx_sorted = tx.sort_values(["sender_iban", "timestamp"]).copy()
    tx_sorted["first_time_recipient"] = (
        tx_sorted.groupby(["sender_iban", "recipient_iban"]).cumcount() == 0
    ).astype(float)

    dup_amt = tx_sorted.groupby(["sender_iban", "recipient_iban", "amount"]).cumcount()
    tx_sorted["repeat_same_amount"] = (dup_amt > 0).astype(float)

    tx_sorted["round_hundred"] = ((tx_sorted["amount"] % 100) == 0).astype(float)
    tx_sorted["is_outbound_customer"] = tx_sorted["sender_is_customer"]

    burst = tx_sorted.groupby("sender_iban", group_keys=False).rolling(
        "24h", on="timestamp", min_periods=1
    )["transaction_id"].count()
    # Rolling result uses a MultiIndex; values align with tx_sorted row order.
    tx_sorted["burst_24h_count"] = pd.to_numeric(burst, errors="coerce").fillna(1.0).to_numpy()
    tx = tx_sorted.sort_index()

    tx_ord = tx.sort_values(["sender_iban", "timestamp"])
    bal = tx_ord.groupby("sender_iban")["balance_after"].pct_change().abs()
    tx_ord = tx_ord.copy()
    tx_ord["balance_shock"] = bal.fillna(0.0).clip(0, 5)
    tx = tx_ord.sort_index()

    msg_df = build_message_risk_tables(mails, sms, users_v)
    ph_scores = []
    ph_flags_col = []
    for _, r in tx.iterrows():
        ps, fl = _window_score(msg_df, r["timestamp"])
        ph_scores.append(ps)
        ph_flags_col.append("|".join(fl))
    tx["phishing_window_score"] = ph_scores
    tx["phishing_flags"] = ph_flags_col

    geo_feats = []
    for _, r in tx.iterrows():
        ib = r.get("sender_iban")
        res_lat = float(lat_map.get(ib, np.nan))
        res_lng = float(lng_map.get(ib, np.nan))
        d_res, d_ping, spd = _nearest_location_features(r, locs, res_lat, res_lng)
        loc_str = str(r.get("location") or "").lower()
        city = str(city_map.get(ib, "")).lower()
        city_mismatch = 0.0
        if loc_str and city and city not in loc_str and loc_str not in ("", "nan"):
            city_mismatch = 0.25 * (1.0 - fuzz.partial_ratio(city, loc_str) / 100.0)
        imp_travel = clip01((spd - 900) / 900) if spd > 900 else 0.0
        dist_anom = clip01((d_res - 50) / 200) if np.isfinite(d_res) else 0.0
        geo_feats.append((dist_anom, imp_travel + city_mismatch))
    tx["geo_distance_anomaly"] = [g[0] for g in geo_feats]
    tx["geo_impossible_or_mismatch"] = [g[1] for g in geo_feats]

    tx["txn_anomaly_score"] = clip01(
        0.22 * np.tanh(tx["amount_z"].abs() / 3)
        + 0.18 * np.tanh(tx["amount_to_monthly_ratio"] / 4)
        + 0.12 * tx["first_time_recipient"]
        + 0.12 * np.tanh((tx["burst_24h_count"] - 3).clip(lower=0) / 5)
        + 0.12 * tx["balance_shock"]
        + 0.10 * tx["round_hundred"]
        + 0.06 * tx["repeat_same_amount"]
        + 0.08 * (~tx["transaction_type"].astype(str).str.lower().isin(["transfer", "e-commerce"])).astype(float)
    )

    tx["geo_score"] = clip01(0.55 * tx["geo_impossible_or_mismatch"] + 0.45 * tx["geo_distance_anomaly"])
    tx["phishing_score"] = clip01(tx["phishing_window_score"] * 0.85 + tx["user_vulnerability"] * 0.15)
    tx["vuln_score"] = tx["user_vulnerability"]

    tx["economic_risk_score"] = clip01(
        0.55 * np.tanh(tx["amount"] / (tx["salary"].replace(0, np.nan) + 1.0))
        + 0.25 * np.tanh(tx["amount_to_monthly_ratio"] / 3)
        + 0.20 * tx["balance_shock"]
    )

    tx["base_risk_score"] = clip01(
        0.34 * tx["txn_anomaly_score"]
        + 0.28 * tx["geo_score"]
        + 0.28 * tx["phishing_score"]
        + 0.10 * tx["vuln_score"]
    )

    reasons = []
    for _, r in tx.iterrows():
        flags: list[str] = []
        if r["txn_anomaly_score"] > 0.55:
            flags.append("txn_anomaly")
        if r["geo_score"] > 0.45:
            flags.append("geo_inconsistency")
        if r["phishing_score"] > 0.45:
            flags.append("phishing_exposure")
        if r["economic_risk_score"] > 0.55:
            flags.append("high_value")
        if r["first_time_recipient"] == 1 and r["amount_to_monthly_ratio"] > 1.5:
            flags.append("new_counterparty_high_ratio")
        reasons.append("|".join(flags))
    tx["reason_flags"] = reasons

    def _priority(row: pd.Series) -> ReviewPriority:
        br = float(row["base_risk_score"])
        eco = float(row["economic_risk_score"])
        ph = float(row["phishing_score"])
        if br > 0.72 or eco > 0.78 or ph > 0.72:
            return "high"
        if br > 0.45 or eco > 0.55 or ph > 0.5:
            return "medium"
        return "low"

    tx["llm_review_priority"] = tx.apply(_priority, axis=1)

    tx["cluster_risk"] = (
        tx.groupby("sender_iban")["base_risk_score"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        .fillna(tx["base_risk_score"])
    )
    for col in [
        "txn_anomaly_score",
        "geo_score",
        "phishing_score",
        "vuln_score",
        "economic_risk_score",
        "base_risk_score",
        "cluster_risk",
        "burst_24h_count",
    ]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0.0 if col != "burst_24h_count" else 1.0)
    return tx
