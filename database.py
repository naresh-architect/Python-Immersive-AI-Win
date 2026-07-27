"""
SupportPilot — mock ShopStream India database.

Creates and seeds a small SQLite database with synthetic customers and
orders, including deliberate edge cases: a customer with no orders, a
flagged (fraud-risk) customer, and a high-value order used to exercise
the refund hard-escalation rule.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "shopstream.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    account_standing TEXT,
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    item TEXT,
    amount REAL,
    status TEXT,
    order_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT,
    created_at TEXT
);
"""

CUSTOMERS = [
    ("CUST001", "Ananya Rao", "ananya.rao@example.com", "good", "2023-01-15"),
    ("CUST002", "Vikram Shah", "vikram.shah@example.com", "vip", "2022-06-02"),
    ("CUST003", "Priya Nair", "priya.nair@example.com", "good", "2023-11-20"),
    ("CUST004", "Rohan Mehta", "rohan.mehta@example.com", "flagged", "2024-02-10"),
    ("CUST005", "Sneha Iyer", "sneha.iyer@example.com", "good", "2023-05-30"),
    ("CUST006", "Arjun Kapoor", "arjun.kapoor@example.com", "good", "2024-01-05"),
    ("CUST007", "Divya Menon", "divya.menon@example.com", "good", "2022-09-18"),
    ("CUST008", "Karan Malhotra", "karan.malhotra@example.com", "vip", "2021-12-01"),
    ("CUST009", "Ritu Desai", "ritu.desai@example.com", "good", "2024-03-22"),
    ("CUST010", "Aditya Joshi", "aditya.joshi@example.com", "good", "2023-08-14"),
    # Edge case: brand new customer, no orders yet
    ("CUST011", "Meera Pillai", "meera.pillai@example.com", "good", "2025-06-01"),
]

_ITEMS = [
    "Wireless Earbuds", "Smart Watch", "Cotton Kurta", "Running Shoes",
    "Bluetooth Speaker", "Laptop Backpack", "Kitchen Blender", "Yoga Mat",
    "LED Desk Lamp", "Air Fryer", "Bedsheet Set", "Sunglasses",
]

ORDERS = []
_order_counter = 1
for cust_id in [c[0] for c in CUSTOMERS if c[0] != "CUST011"]:
    n_orders = 5 if cust_id != "CUST002" else 1  # CUST002 gets one high-value order
    for i in range(n_orders):
        oid = f"ORD{_order_counter:04d}"
        item = _ITEMS[_order_counter % len(_ITEMS)]
        amount = 6500.0 if cust_id == "CUST002" else round(300 + (_order_counter * 137) % 4000, 2)
        status = ["delivered", "in_transit", "delivered", "returned", "cancelled"][i % 5]
        ORDERS.append((oid, cust_id, item, amount, status, "2025-0" + str((i % 6) + 1) + "-10"))
        _order_counter += 1


def init_db(force: bool = False) -> None:
    if force and DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?)", CUSTOMERS
    )
    conn.executemany(
        "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)", ORDERS
    )
    conn.commit()
    conn.close()


def get_customer(customer_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_orders(customer_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC", (customer_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prior_ticket_count(customer_id: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else 0


if __name__ == "__main__":
    init_db(force=True)
    print(f"Seeded database at {DB_PATH}")
    print(f"Customers: {len(CUSTOMERS)} | Orders: {len(ORDERS)}")
