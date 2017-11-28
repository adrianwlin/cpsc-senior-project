import sqlite3
import sys
import pickle
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


def label_data(pickle_file, outfile, connection):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    with open(pickle_file, "r") as f:
        labeled_list = pickle.load(f)
        # sentences with both genes and disease tags
        good_sentences = []
        no_good = 0
        good = 0
        positive = 0
        negative = 0
        for entry in labeled_list:
            if len(entry["genes"]) == 0 or len(entry["diseases"]) == 0:
                no_good += 1
            else:
                good += len(entry["genes"]) * len(entry["diseases"])
                for gene in entry["genes"]:
                    uniprot = gene["uniprot"]
                    # if the uniprot ID is blank
                    if not uniprot:
                        continue
                    cur = connection.cursor()
                    cur.execute(
                        "SELECT geneNID, geneName FROM geneAttributes WHERE uniprotId = \"{}\"".format(uniprot))
                    rows = cur.fetchall()
                    # If we cannot find this gene in the db, continue
                    if len(rows) != 1:
                        continue
                    geneNID, geneName = rows[0]

                    for disease in entry["diseases"]:
                        cui = disease["cui"]
                        cur = connection.cursor()
                        cur.execute(
                            "SELECT diseaseNID, diseaseName FROM diseaseAttributes WHERE diseaseId = \"{}\"".format(cui))
                        rows = cur.fetchall()
                        # If we cannot find this disease in the db, continue
                        if len(rows) != 1:
                            continue
                        diseaseNID, diseaseName = rows[0]

                        cur = connection.cursor()
                        cur.execute(
                            "SELECT associationType FROM geneDiseaseNetwork WHERE geneNID = \"{}\" AND diseaseNID = \"{}\"".format(geneNID, diseaseNID))
                        rows = cur.fetchall()
                        # If the association exists, label true
                        if len(rows) == 1:
                            entry.update(
                                {"labels": {(gene["name"], disease["name"]): True}})
                            positive += 1
                        else:
                            entry.update(
                                {"labels": {(gene["name"], disease["name"]): False}})
                            negative += 1
                # If some gene/disease pair has been labeled, add it to good_sentences.
                if "labels" in entry:
                    good_sentences.append(entry.copy())
        print len(good_sentences), "good sentences"
        print no_good, good
        print "positive: {}, negative {}".format(positive, negative)

        f = open(outfile, 'wb')
        pickle.dump(labeled_list, f)
        f.close()


def main():
    """cytoscape_lookup.py [db_file] [pickle_file] [outfile]
    """
    # Example code for reading pickle file
    # f = open(pickleDumpFile,'r')
    # testLoad = pickle.load(f)
    # print(testLoad == entityList)
    if len(sys.argv) != 4:
        print "cytoscape_lookup.py [db_file] [pickle_file] [outfile]"
        return 1
    else:
        db_file = sys.argv[1]
        pickle_file = sys.argv[2]
        outfile = sys.argv[3]
        connection = create_connection(db_file)
        with connection:
            label_data(pickle_file, outfile, connection)
    return 0


if __name__ == "__main__":
    main()
