# Category expenses prediction REST API service

## HOWTO:
### 1. Serialize data:
Make sure you moved all .csv files to `cache/tinkoff_hackathon_data`

`python tools/seialize_data.py`

### 2. Deserialize data:

With `parse_file_as` from `pydantic.tools`:

`parse_file_as(List[UserDataModel], 'data/json/users_test.json')`