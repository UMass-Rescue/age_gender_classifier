### How to get the evaluation data set
The evaluation dataset is 199MB in size, too large to check into GitHub.  
Download it from [here](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv/data), and save the file in this directory as `age_gender.csv`.  
Then run the `formatter.py` file; this runs some minor processing and creates a new file, `age_gender_indexed.csv`.  

### Load into PostgreSQL

If running a local postgres instance with `bash spin-up-local-db.sh`, you can quickly copy the CSV file into PostgreSQL using the following commands. Note that the relation must exist before running the `copy` command.  

    docker exec -it cs596 bash
    psql -U postgres -d age_gender_labeled
    copy age_gender_labeled from '/tmp/postgres/age_gender.csv' delimiter ',' csv;