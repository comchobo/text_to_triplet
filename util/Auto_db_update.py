def connect_DB(table_name='embargo'): # company owns DB
    import oracledb
    oracledb.init_oracle_client(lib_dir=r"C:\oracle")

    host = 'blinded' # company owns DB
    port = 'blinded' # company owns DB
    service_name = 'blinded' # company owns DB
    username = 'blinded' # company owns DB
    password = 'blinded' # company owns DB
    connection = oracledb.connect(user=username, password=password, host=host, port=port,
                                  service_name=service_name)

    sql = f"SELECT datetime FROM {table_name} ORDER BY datetime DESC FETCH FIRST 200 ROWS ONLY"
    cursor = connection.cursor()
    most_current_in_db = cursor.execute(sql).fetchall()
    most_current_in_db = [x[0] for x in most_current_in_db]
    most_current_in_db = list(set(most_current_in_db))
    if 'nan' in most_current_in_db : most_current_in_db.remove('nan')
    if None in most_current_in_db: most_current_in_db.remove(None)
    return connection, most_current_in_db[0]


def fetch_from_DB(most_current_in_db, most_current_in_local, connection, target_table, save_path):
    import pandas as pd

    from datetime import date, timedelta, datetime
    start_date = datetime.strptime(str(most_current_in_local), '%Y%m%d').date()
    end_date = datetime.strptime(most_current_in_db, '%Y%m%d').date()
    delta = timedelta(days=1)
    date_range = []

    while start_date <= end_date:
        date_range.append(start_date.strftime("%Y%m%d"))
        start_date += delta

    if target_table == 'blinded': # company owns DB
        column_names = 'blinded' # company owns DB
    else:
        column_names = 'blinded' # company owns DB

    from tqdm import tqdm
    for date_val in tqdm(date_range):
        sql = f"SELECT * FROM {target_table} WHERE datetime='{date_val}'"
        cursor = connection.cursor()
        res = cursor.execute(sql).fetchall()
        pd.DataFrame(res, columns=column_names).to_csv(f'{save_path}/{date_val}.csv')
    return date_range


def check_update(target_path):
    print('checking update of db in during lunch')
    target_dict_list =[{'table':'blinded','path':f'{target_path}/new_korean_news'} company# company owns DB
        ,{'table':'blinded','path':f'{target_path}/new_english_news'}] # company owns DB

    import os
    for target_dict in target_dict_list:
        connection, most_current_in_db = connect_DB(target_dict['table'])
        temp = os.listdir(target_dict['path'])
        temp.sort()
        most_current_in_local = int(temp[-1][:-4])
        if int(most_current_in_db) > most_current_in_local:
            print('\nfetching new data...')
            fetch_from_DB(most_current_in_db, most_current_in_local, connection,
                          target_dict['table'], target_dict['path'])
        else:
            print(f"\n{target_dict['table']} data is up-to-date.")
    print('gonna sleep till tomorrow...')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", default='d:/dataset')
    args = parser.parse_args()
    check_update(args.target_path)
    import schedule
    schedule.every().day.at("12:00").do(check_update, args.target_path)

    import time
    while True:
        schedule.run_pending()
        time.sleep(3600)
