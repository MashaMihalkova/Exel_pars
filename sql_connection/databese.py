from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

# SQLALCHEMY_DATABASE_URL = "postgresql://osm:geoserver@10.0.1.64:5432/db_etl_pm_pmc"
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:34uzisup@localhost:5432/db_test"
SQLALCHEMY_DATABASE_URL = "postgresql://PMAdmin:PMAdmin@sql2019d01.cs.local:1433/ASU_GSP_dev"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

