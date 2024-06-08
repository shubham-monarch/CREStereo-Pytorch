import coloredlogs, logging

# testing 3 channel np arrs
def test_arr(arr, str =""):
    logging.debug(f"Testing {str} ..")
    logging.debug(f"Shape: {arr.shape} dtype: {arr.dtype}")    
    logging.debug(f"{arr[:3, :3]}")