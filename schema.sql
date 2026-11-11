-- Tabelas de dimensão primeiro (fato_preco referencia as duas via FK,
-- então precisam existir antes dela ser criada).

CREATE TABLE dim_empresa (
    ticker VARCHAR(10) PRIMARY KEY,
    empresa VARCHAR(100),
    setor VARCHAR(50),
    pais VARCHAR(50),
    market_cap FLOAT
);

CREATE TABLE dim_tempo (
    data DATE PRIMARY KEY,
    ano INT,
    mes INT,
    trimestre INT
);

CREATE TABLE fato_preco (
    data DATE,
    ticker VARCHAR(10),
    abertura FLOAT,
    maxima FLOAT,
    minima FLOAT,
    fechamento FLOAT,
    volume BIGINT,
    retorno_diario FLOAT,
    PRIMARY KEY (data, ticker),
    FOREIGN KEY (ticker) REFERENCES dim_empresa(ticker),
    FOREIGN KEY (data) REFERENCES dim_tempo(data)
);
