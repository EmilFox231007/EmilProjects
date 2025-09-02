## Installation

To install the `axonrooter` package from the built `.whl` file, follow the steps below:

### 1. Build the wheel package

If not already built, you can create the `.whl` using Poetry:

```bash
poetry build
```

### 2. Install the package

Activate the cv_env Conda environment and run:

```bash
pip install dist/axonrooter-0.1.0-py3-none-any.whl
```

Conda environment can be set up using Conda environment set-up guide in the Other Evidence/Setups and Guides dir of this repo.

### 3. Verify installation

You can verify it installed correctly by trying to import the module:

```bash
python -c "import axonrooter; print(axonrooter.__version__)"
```

## API Usage

### Expected input and output

**Input:**

A single image file (`.png`, `.jpg`, `.jpeg`) sent as a `POST` request to the `/predict/` endpoint.

**Output:**

A JSON object containing a list of root tip coordinates (x, y and up to 5 roots total), based on model predictions.

**Example Output:**

```json
{
  "tips": [
    [104, 2932],
    [210, 2875],
    [412, 2890]
  ]
}
```

### Example python requests request

Full example of how to get a JSON response with a local test image as the selected image, and python requests library handling the request.

```python
import requests

# The URL of the API
api_url = 'http://localhost:8000/'
endpoint = 'predict/'
url = api_url + endpoint

# Create the payload
payload = {'path_to_img': f"C:/Users/Endijs/Desktop/School/Y2/Block D/Testing images/test1.png"}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())
```

### **Status Code**

* `200 OK`: If prediction is successful.
* `422 Unprocessable Entity`: If the uploaded file is missing or corrupted.
* `500 Internal Server Error`: If inference fails internally.


## Logging

This project uses Python’s built-in logging module to track application events. Logging is configured to provide useful information during development and production, and is easy to adjust based on your environment.

### Output logs file
The logs are saved in `src/axonrooter/logs/app.log`, logs include timestamps, log level, module name, and the message for easier debugging and tracing.

### Debug

For debug usage, access the `logging_config.py` under `src/axonrooter`, and change the config parameter to DEBUG.

### Example:
```python
from config import setup_logging
import logging

setup_logging(level=logging.DEBUG)

```
**This will**:

- Show detailed debug messages in both the console and logs/app.log.

- Help trace the full internal behavior of the app.

### **Note**: Common log levels include:

**DEBUG**: Most verbose, useful for development.

**INFO**: General application flow.

**WARNING**: Something unexpected, but not critical.

**ERROR**: A problem that needs attention.

**CRITICAL**: Major failure.


### Enable/Disable Logging for Production
If you want to silently run the app without any logs, you can disable all the loggings by changing the following code at the top of the file `src/axonrooter/app/app.py`:
```python
from axonrooter.logging_config import configure_logging

# Enable logging at DEBUG level (for dev)
'''comment this out'''
# configure_logging(enabled=True, level=logging.DEBUG)

# Or disable completely (for production)
'''uncomment this'''
configure_logging(enabled=False)

```
