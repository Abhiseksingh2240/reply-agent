import pandas as pd

from app.scoring import score_transactions


def test_geo_scoring_smoke():
    tx = pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "sender_id": "BIO-1",
                "recipient_id": "X",
                "transaction_type": "transfer",
                "amount": 500.0,
                "location": "",
                "payment_method": "iban",
                "sender_iban": "IBAN1",
                "recipient_iban": "IBAN2",
                "balance_after": 1000.0,
                "description": "",
                "timestamp": pd.Timestamp("2087-01-10T12:00:00Z"),
            }
        ]
    )
    users = pd.DataFrame(
        [
            {
                "first_name": "A",
                "last_name": "B",
                "birth_year": 2000,
                "salary": 12000,
                "job": "clerk",
                "iban": "IBAN1",
                "residence_city": "Home",
                "residence_lat": 0.0,
                "residence_lng": 0.0,
                "description": "low phishing risk",
                "biotag": "BIO-1",
            }
        ]
    )
    locs = pd.DataFrame(
        [
            {"biotag": "BIO-1", "timestamp": pd.Timestamp("2087-01-10T11:00:00Z"), "lat": 0.01, "lng": 0.01, "city": "Home"}
        ]
    )
    mails = pd.DataFrame(columns=["mail_timestamp", "subject", "body", "to_addr", "from_addr"])
    sms = pd.DataFrame(columns=["sms_timestamp", "sms_text", "sms_from", "sms_to"])
    audio = pd.DataFrame(columns=["path", "name", "suffix", "size_bytes"])
    tables = {
        "transactions": tx,
        "users": users,
        "locations": locs,
        "mails": mails,
        "sms": sms,
        "audio": audio,
    }
    out = score_transactions(tables)
    assert "geo_score" in out.columns
    assert out["geo_score"].iloc[0] >= 0.0
