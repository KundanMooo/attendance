import sqlite3

class Database:
    def __init__(self, db_name='face_detection.db'):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.initialize_database()

    def initialize_database(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fingerprint TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def insert_visit(self, fingerprint):
        self.cursor.execute('''
            INSERT INTO visits (fingerprint) VALUES (?)
        ''', (fingerprint,))
        self.connection.commit()

    def check_fingerprint_exists(self, fingerprint):
        self.cursor.execute('''
            SELECT * FROM visits WHERE fingerprint = ?
        ''', (fingerprint,))
        return self.cursor.fetchone() is not None

    def close(self):
        self.connection.close()