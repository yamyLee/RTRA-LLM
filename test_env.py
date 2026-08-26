#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()
print('DEEPSEEK_API_KEY:', 'set' if os.environ.get('DEEPSEEK_API_KEY') else 'NOT SET')
print('DEEPSEEK_BASE_URL:', os.environ.get('DEEPSEEK_BASE_URL', 'NOT SET'))
print('LLM_PROVIDER:', os.environ.get('LLM_PROVIDER', 'NOT SET'))
