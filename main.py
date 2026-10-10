import csv

with open("data/bugs.csv", "r") as file:
    bugs = csv.DictReader(file)

    for bug in bugs:
        print(bug["id"], "-", bug["title"])
