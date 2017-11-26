import sqlite3
import sys
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


def query(connection):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = connection.cursor()
    cur.execute("SELECT * FROM geneAttributes LIMIT 10")

    rows = cur.fetchall()

    for row in rows:
        print(row)


def main():
    """cytoscape_lookup.py [db_file]
    """
    if len(sys.argv) != 2:
        print "cytoscape_lookup.py [db_file]"
        return 1
    else:
        db_file = sys.argv[1]
        connection = create_connection(db_file)
        with connection:
            query(connection)
    return 0


if __name__ == "__main__":
    main()
