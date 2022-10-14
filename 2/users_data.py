import pandas as pd

events_data = pd.read_csv('events_data_test.csv')
submission_data = pd.read_csv('submission_data_test.csv')

events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date

submission_data['date'] = pd.to_datetime(submission_data.timestamp, unit='s')
submission_data['day'] = submission_data.date.dt.date

users_data = events_data.groupby('user_id', as_index=False).agg({'timestamp': 'max'}).rename({'timestamp'
                                                                                              : 'last_timestamp'},
                                                                                             axis='columns')
now = 1526772811
drop_out_threasold = 2592000

users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threasold
users_scores = submission_data.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()

users_data = users_data.merge(users_scores, on='user_id', how='outer')
users_data = users_data.fillna(0)
users_invent_data = events_data.pivot_table(index='user_id',
                                            columns='action',
                                            values='step_id',
                                            aggfunc='count',
                                            fill_value=0).reset_index()
users_data = users_data.merge(users_invent_data, how='outer')
users_days = events_data.groupby('user_id').day.nunique()
users_days.to_frame().reset_index()
users_data = users_data.merge(users_days, on='user_id', how='outer')
users_data['passed_corse'] = users_data.passed > 170

user_min_time = events_data.groupby('user_id', as_index=False).agg({'timestamp'
                                                                    : 'min'}).rename({'timestamp': 'min_timestamp'},
                                                                                     axis=1)
users_data = users_data.merge(user_min_time, how='outer')

event_data_train = events_data[
    events_data.timestamp <= events_data.
    merge(user_min_time, on='user_id', how='left')['min_timestamp'] + 3*24*60*60]

print(event_data_train)
