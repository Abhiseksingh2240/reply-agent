"""Normalize raw tables into unified snake_case dataframes."""

from __future__ import annotations

import email
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.io_loader import RawDatasetBundle
from app.utils import is_employer_id, looks_like_biotag, parse_iso_utc, snake_case_columns


def _extract_email_address(header_val: str) -> str:
    m = re.search(r"<([^>]+)>", header_val or "")
    if m:
        return m.group(1).strip().lower()
    return (header_val or "").strip().lower()


def _parse_rfc_mail(raw: str) -> dict[str, Any]:
    msg = email.message_from_string(raw)
    subj = msg.get("Subject", "") or ""
    from_ = msg.get("From", "") or ""
    to = msg.get("To", "") or ""
    date_hdr = msg.get("Date", "") or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    body += str(part.get_payload())
            elif ctype == "text/html" and not body:
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    body += str(part.get_payload())
    else:
        try:
            body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            body = str(msg.get_payload())
    ts = pd.NaT
    try:
        dt = email.utils.parsedate_to_datetime(date_hdr)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ts = pd.Timestamp(dt).tz_convert("UTC")
    except Exception:
        pass
    return {
        "from_addr": _extract_email_address(from_),
        "to_addr": _extract_email_address(to),
        "subject": subj,
        "body": body[:200_000],
        "mail_timestamp": ts,
        "raw": raw[:5000],
    }


def _parse_sms_block(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "=== SMS CONVERSATION ===" in raw:
        parts = re.split(r"From:", raw)
        for chunk in parts[1:]:
            chunk = "From:" + chunk
            m_from = re.search(r"From:\s*(.+)", chunk)
            m_to = re.search(r"To:\s*(.+)", chunk)
            m_date = re.search(r"Date:\s*(.+)", chunk)
            m_msg = re.search(r"Message:\s*(.+)", chunk, re.DOTALL)
            ts = pd.NaT
            if m_date:
                try:
                    ts = pd.to_datetime(m_date.group(1).strip(), utc=True, errors="coerce")
                except Exception:
                    ts = pd.NaT
            rows.append(
                {
                    "sms_from": (m_from.group(1).split("\n")[0].strip() if m_from else ""),
                    "sms_to": (m_to.group(1).split("\n")[0].strip() if m_to else ""),
                    "sms_timestamp": ts,
                    "sms_text": (m_msg.group(1).strip() if m_msg else chunk)[:20_000],
                }
            )
        return rows

    m_from = re.search(r"From:\s*(.+)", raw)
    m_to = re.search(r"To:\s*(.+)", raw)
    m_date = re.search(r"Date:\s*(.+)", raw)
    m_msg = re.search(r"Message:\s*(.+)", raw, re.DOTALL)
    ts = pd.NaT
    if m_date:
        ts = pd.to_datetime(m_date.group(1).strip(), utc=True, errors="coerce")
    rows.append(
        {
            "sms_from": (m_from.group(1).split("\n")[0].strip() if m_from else ""),
            "sms_to": (m_to.group(1).split("\n")[0].strip() if m_to else ""),
            "sms_timestamp": ts,
            "sms_text": (m_msg.group(1).strip() if m_msg else raw)[:20_000],
        }
    )
    return rows


def build_user_tables(users: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for u in users:
        res = u.get("residence") or {}
        rows.append(
            {
                "first_name": u.get("first_name"),
                "last_name": u.get("last_name"),
                "birth_year": u.get("birth_year"),
                "salary": float(u.get("salary") or 0.0),
                "job": u.get("job"),
                "iban": str(u.get("iban") or "").upper().replace(" ", ""),
                "residence_city": (res.get("city") or ""),
                "residence_lat": float(res.get("lat") or np.nan),
                "residence_lng": float(res.get("lng") or np.nan),
                "description": u.get("description") or "",
            }
        )
    return pd.DataFrame(rows)


def infer_biotag_for_iban(transactions: pd.DataFrame) -> dict[str, str]:
    """Map personal IBAN -> likely biotag from salary credits."""
    iban_to_bio: dict[str, str] = {}
    for _, r in transactions.iterrows():
        sid = str(r.get("sender_id", ""))
        if is_employer_id(sid):
            rec_iban = str(r.get("recipient_iban", "")).upper().replace(" ", "")
            rid = str(r.get("recipient_id", ""))
            if rec_iban and looks_like_biotag(rid):
                iban_to_bio[rec_iban] = rid
    return iban_to_bio


def normalize_bundle(bundle: RawDatasetBundle) -> dict[str, Any]:
    tx = snake_case_columns(bundle.transactions.copy())
    if "timestamp" in tx.columns:
        tx["timestamp"] = parse_iso_utc(tx["timestamp"])
    for col in ["amount", "balance_after"]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")
    for col in ["sender_iban", "recipient_iban"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.upper().str.replace(" ", "", regex=False)

    users_df = build_user_tables(bundle.users)
    iban_to_bio = infer_biotag_for_iban(tx)
    users_df["biotag"] = users_df["iban"].map(lambda ib: iban_to_bio.get(ib))

    loc_rows = []
    for loc in bundle.locations:
        loc_rows.append(
            {
                "biotag": loc.get("biotag"),
                "timestamp": parse_iso_utc(pd.Series([loc.get("timestamp")])).iloc[0],
                "lat": float(loc.get("lat")),
                "lng": float(loc.get("lng")),
                "city": loc.get("city"),
            }
        )
    locations_df = pd.DataFrame(loc_rows)

    mail_rows = []
    for item in bundle.mails:
        raw = item.get("mail") or item.get("body") or ""
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = _parse_rfc_mail(raw)
        mail_rows.append(parsed)
    mails_df = pd.DataFrame(mail_rows)

    sms_rows: list[dict[str, Any]] = []
    for item in bundle.sms:
        raw = item.get("sms") or ""
        if not isinstance(raw, str):
            raw = str(raw)
        sms_rows.extend(_parse_sms_block(raw))
    sms_df = pd.DataFrame(sms_rows)

    audio_df = pd.DataFrame(
        [
            {
                "path": str(p),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for p in bundle.audio_files
        ]
    )

    return {
        "transactions": tx,
        "users": users_df,
        "locations": locations_df,
        "mails": mails_df,
        "sms": sms_df,
        "audio": audio_df,
        "data_dir": bundle.data_dir,
    }
