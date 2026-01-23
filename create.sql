CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    income DECIMAL(18,2),
    amount DECIMAL(18,2) NOT NULL,
    ts DATETIME2 NOT NULL,
    description NVARCHAR(500)
);