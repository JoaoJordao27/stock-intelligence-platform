"""
Stock Intelligence Platform - Etapa 3: Carga no Banco
=========================================================
Cria o banco a partir de schema.sql e carrega os 3 CSVs gerados na Etapa 1
(output/fato_preco.csv, output/dim_empresa.csv, output/dim_tempo.csv).

MOTOR DE BANCO: SQLite (`stock_intelligence.db`, arquivo local neste
diretório). Escolhido por ser o padrão mais simples para um projeto local
sem servidor de banco configurado — zero setup, só o módulo `sqlite3` da
biblioteca padrão do Python. `schema.sql` é SQL padrão (CREATE TABLE sem
nada específico de motor) e roda sem alterações no SQLite; se o projeto
migrar para Postgres/MySQL depois, o mesmo schema.sql serve de base (só
trocar a camada de conexão neste script, de `sqlite3` para
`psycopg2`/`mysql-connector`, por exemplo).

O script é IDEMPOTENTE: roda `DROP TABLE IF EXISTS` antes de recriar, então
pode ser executado várias vezes (ex. depois de rodar a Etapa 1 de novo com
dados atualizados) sem duplicar linhas nem falhar por tabela já existente.

CHAVES: `schema.sql` agora declara PRIMARY KEY em cada tabela — (data,
ticker) em fato_preco, ticker em dim_empresa, data em dim_tempo — e FOREIGN
KEY de fato_preco pra dim_empresa (ticker) e dim_tempo (data). Isso muda a
ORDEM em que as tabelas precisam ser criadas/carregadas/dropadas:
  - CRIAR e CARREGAR: dimensões primeiro (dim_empresa, dim_tempo), fato por
    último — senão o SQLite recusa a FK por apontar pra tabela/linha que
    ainda não existe.
  - DROPAR: ordem inversa, fato primeiro — senão seria preciso dropar uma
    tabela referenciada por FK antes da que referencia ela.
  `PRAGMA foreign_keys = ON` é ativado logo na conexão — o SQLite, ao
  contrário da maioria dos bancos, IGNORA declarações de FOREIGN KEY por
  padrão a menos que essa pragma seja ligada explicitamente por conexão.

Uso:
    python 03_carga_banco.py
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "schema.sql"
OUTPUT_DIR = HERE / "output"
DB_PATH = HERE / "stock_intelligence.db"

# Ordem de dependência: fato_preco tem FK pras duas dimensões.
TABELAS_DIMENSAO = ["dim_empresa", "dim_tempo"]
TABELA_FATO = "fato_preco"
ORDEM_CARGA = TABELAS_DIMENSAO + [TABELA_FATO]   # pais antes do filho
ORDEM_DROP = [TABELA_FATO] + TABELAS_DIMENSAO    # filho antes dos pais

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("carga_banco")


def recriar_schema(conn: sqlite3.Connection):
    """Apaga as tabelas (se existirem, filho antes dos pais por causa das
    FKs) e recria a partir de schema.sql — torna o script seguro pra rodar
    de novo sem erro de 'table already exists' nem duplicar dados de uma
    carga anterior."""
    cursor = conn.cursor()
    for tabela in ORDEM_DROP:
        cursor.execute(f"DROP TABLE IF EXISTS {tabela}")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    cursor.executescript(schema_sql)
    conn.commit()
    log.info(f"Schema recriado a partir de {SCHEMA_PATH.name} ({len(ORDEM_CARGA)} tabelas, com PK/FK).")


def carregar_tabela(conn: sqlite3.Connection, nome_tabela: str):
    caminho_csv = OUTPUT_DIR / f"{nome_tabela}.csv"
    df = pd.read_csv(caminho_csv)
    df.to_sql(nome_tabela, conn, if_exists="append", index=False)
    log.info(f"{nome_tabela}: {len(df)} linhas carregadas de {caminho_csv.name}")


def testar_integridade_referencial(conn: sqlite3.Connection):
    """Tenta inserir uma linha em fato_preco com um ticker que NÃO existe em
    dim_empresa — prova que a FK está de fato sendo aplicada pelo SQLite
    (não só declarada no schema). O INSERT deve falhar com IntegrityError;
    se passar, é sinal de que PRAGMA foreign_keys não está ativo."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO fato_preco (data, ticker, abertura, maxima, minima, fechamento, volume, retorno_diario) "
            "VALUES ('2099-01-01', 'TICKER_INEXISTENTE', 1, 1, 1, 1, 1, 0)"
        )
        conn.rollback()
        log.warning("FALHA NA VALIDAÇÃO: o INSERT com ticker inválido foi aceito — "
                    "a FK não está sendo aplicada (verifique PRAGMA foreign_keys).")
    except sqlite3.IntegrityError as e:
        conn.rollback()
        log.info(f"Integridade referencial OK — INSERT com ticker inválido foi corretamente "
                 f"rejeitado: {e}")


def checagens_pos_carga(conn: sqlite3.Connection):
    """Confere contagens e roda uma query de exemplo que só faz sentido com
    as 3 tabelas já relacionadas — prova de que a carga funcionou de forma
    utilizável, não só que os INSERTs não deram erro."""
    cursor = conn.cursor()

    log.info("--- Contagem de linhas por tabela ---")
    for tabela in ORDEM_CARGA:
        n = cursor.execute(f"SELECT COUNT(*) FROM {tabela}").fetchone()[0]
        log.info(f"  {tabela}: {n} linhas")

    log.info("--- Query de exemplo: retorno médio por setor (join fato_preco x dim_empresa) ---")
    query = """
        SELECT
            e.setor,
            COUNT(*) AS n_observacoes,
            ROUND(AVG(f.retorno_diario) * 100, 4) AS retorno_medio_diario_pct,
            ROUND(SUM(f.volume), 0) AS volume_total
        FROM fato_preco f
        JOIN dim_empresa e ON f.ticker = e.ticker
        GROUP BY e.setor
        ORDER BY retorno_medio_diario_pct DESC
    """
    resultado = pd.read_sql_query(query, conn)
    print(resultado.to_string(index=False))

    log.info("--- Query de exemplo: preço mais recente por ticker (join fato_preco x dim_tempo) ---")
    query2 = """
        SELECT f.ticker, f.data, f.fechamento
        FROM fato_preco f
        JOIN dim_tempo t ON f.data = t.data
        WHERE f.data = (SELECT MAX(data) FROM fato_preco)
        ORDER BY f.ticker
    """
    resultado2 = pd.read_sql_query(query2, conn)
    print(resultado2.to_string(index=False))


def main():
    log.info(f"Conectando em {DB_PATH.name} (SQLite)")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignora FKs sem isso
    try:
        recriar_schema(conn)
        for tabela in ORDEM_CARGA:
            carregar_tabela(conn, tabela)
        conn.commit()
        checagens_pos_carga(conn)
        testar_integridade_referencial(conn)
    finally:
        conn.close()

    log.info(f"Carga concluída. Banco salvo em: {DB_PATH}")


if __name__ == "__main__":
    main()
