# n8n Booking Agent

This repository contains a sample n8n workflow (`workflow.json`) and basic Python utility used for testing. The workflow implements an email-based flight booking assistant with memory and logging.

## Requirements

* n8n v1 or later
* Google account with Gmail and Google Sheets API access
* OpenAI API key
* Data Store feature enabled in n8n

## Environment Variables

Set the following environment variables inside your n8n instance:

- `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` – for Gmail and Sheets nodes
- `OPENAI_API_KEY` – used by the OpenAI node

## Workflow Overview

1. **Gmail Trigger** – fires when a new email arrives.
2. **Get Memory** – retrieves existing conversation state from n8n Data Store using the thread ID.
3. **Decide Questions** – a Function node that extracts known fields and determines which of the four required questions still need to be asked:
   - travel date
   - destination
   - insurance (yes/no)
   - number of passengers
4. **OpenAI** – generates polite follow‑up questions or a summary for confirmation.
5. **Send Reply** – replies to the email thread with the AI generated message.
6. **Update Memory** – upserts the collected fields back into the Data Store keyed by the thread ID.
7. **Log** – appends a JSON record to a Google Sheet with thread id, sender, extracted fields, status and timestamp.

The assistant never suggests hotels, transfers or pricing, and only asks the four required questions. When all fields are collected it sends a summary and awaits confirmation.

## Logs

The Google Sheet should have the columns:

`thread_id | email_from | extracted_fields | status | timestamp | error`

## Running

1. Import `workflow.json` into your n8n instance.
2. Configure Gmail, Google Sheets and OpenAI credentials.
3. Start the workflow. Send test emails as described below.

### Test Cases

1. **Complete information** – an email that already contains date, destination, insurance preference and passenger count. The agent replies with a summary.
2. **Partial information** – the email has some fields; the agent asks for the missing ones.
3. **Noisy email** – the agent extracts what it can and politely asks only for missing fields.

## Python Utility

`main.py` is an unrelated helper script for checking OpenAI API keys. It is kept for completeness and does not interact with the n8n workflow.

## Development

Run `python -m py_compile main.py` to ensure the Python file is syntactically correct. The workflow JSON can be validated with `jq . workflow.json`.

