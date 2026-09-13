import lancedb
from lancedb.pydantic import LanceModel, Vector

class Schema(LanceModel):
    id: str
    label: str
    confidence: float
    parent_id: str
    caption: str
    image_embeddings: Vector(512)
    text_embeddings: Vector(512)

class LanceDB:
    def __init__(self, table_name="temp1"):
        self.lancedb = lancedb.connect("./lancedb")
        self.table_name = table_name
        self.table = None

    @staticmethod
    def _normalize_record(record):
        if not isinstance(record, dict):
            return record

        for key in ("id", "parent_id"):
            if key in record and record[key] is not None:
                record[key] = str(record[key])

        for key in ("label", "caption"):
            if key in record and record[key] is not None:
                record[key] = str(record[key])

        for key in ("image_embedding", "text_embedding"):
            if key in record and isinstance(record[key], (list, tuple)):
                values = record[key]
                if values and isinstance(values[0], (list, tuple)):
                    values = values[0]
                record[key] = [float(v) for v in values]

        return record

    def insert(self, data):
        normalized_data = [self._normalize_record(item) for item in data]
        available_tables = self.lancedb.list_tables().tables
        if self.table_name not in available_tables:
            self.table = self.lancedb.create_table(
                self.table_name,
                data=normalized_data,
                schema=Schema,
            )
        else:
            self.table = self.lancedb.open_table(self.table_name)

        self.table.add(normalized_data)
