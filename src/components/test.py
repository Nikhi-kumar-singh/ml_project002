try:
    from src.logger import logging
    from src.exception import CustomException
    print("imported successfully")
except Exception as e:
    print(e)