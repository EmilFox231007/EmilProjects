Installation
============

To install the ``axonrooter`` package from the built ``.whl`` file, follow the steps below:

1. Build the wheel package
--------------------------

If not already built, you can create the ``.whl`` using Poetry:

.. code-block:: bash

   poetry build

2. Install the package
----------------------

Activate the `cv_env` Conda environment and run:

.. code-block:: bash

   pip install dist/axonrooter-0.1.0-py3-none-any.whl

The Conda environment can be set up using the Conda setup guide in the "Other Evidence/Setups and Guides" directory of this repo.

3. Verify installation
----------------------

You can verify it installed correctly by trying to import the module:

.. code-block:: bash

   python -c "import axonrooter; print(axonrooter.__version__)"

API Usage
=========

Expected input and output
-------------------------

**Input:**
A single image file (``.png``, ``.jpg``, ``.jpeg``) sent as a ``POST`` request to the ``/predict/`` endpoint.

**Output:**
A JSON object containing a list of root tip coordinates (x, y), up to 5 roots total.

**Example Output:**

.. code-block:: json

   {
     "tips": [
       [104, 2932],
       [210, 2875],
       [412, 2890]
     ]
   }

Example Python requests call
----------------------------

A full example using Python's `requests` library:

.. code-block:: python

   import requests

   api_url = 'http://localhost:8000/'
   endpoint = 'predict/'
   url = api_url + endpoint

   payload = {'path_to_img': "C:/Users/Endijs/Desktop/School/Y2/Block D/Testing images/test1.png"}

   response = requests.post(url, json=payload)

   print(response.json())

Status Codes
------------

- ``200 OK``: Prediction successful
- ``422 Unprocessable Entity``: Missing or corrupted input file
- ``500 Internal Server Error``: Inference failure

Logging
=======

This project uses Python’s built-in logging module to track events.

Output log file
---------------

Logs are saved in ``src/axonrooter/logs/app.log`` and include timestamps, log level, module name, and message.

Enable debug mode
-----------------

To enable debug logging, modify `logging_config.py` in ``src/axonrooter``:

.. code-block:: python

   from config import setup_logging
   import logging

   setup_logging(level=logging.DEBUG)

This will:

- Show detailed debug messages in both the console and log file
- Help trace internal app behavior

Log Levels
----------

- **DEBUG**: Most verbose, useful for development
- **INFO**: General application flow
- **WARNING**: Unexpected but non-fatal
- **ERROR**: Needs attention
- **CRITICAL**: Major failure

Disable logging for production
------------------------------

To silence logging completely, modify the top of ``src/axonrooter/app/app.py``:

.. code-block:: python

   from axonrooter.logging_config import configure_logging

   # Enable for development
   # configure_logging(enabled=True, level=logging.DEBUG)

   # Disable for production
   configure_logging(enabled=False)
