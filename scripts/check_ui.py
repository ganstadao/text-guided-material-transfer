"""Brief local HTTP check for the optional Gradio interface."""
import json
import os
import subprocess
import sys
import time
import urllib.request


def main():
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED='False', NO_PROXY='127.0.0.1,localhost')
    process = subprocess.Popen([sys.executable, 'demo_gradio.py', '--port', '17860'], env=env)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError('UI exited during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:17860/config', timeout=2) as response:
                    config = json.load(response)
                assert config['components'] and config['dependencies']
                with urllib.request.urlopen('http://127.0.0.1:17860/gradio_api/info', timeout=2) as response:
                    info = json.load(response)
                assert info['named_endpoints']
                print('Gradio HTTP/config/API schema OK', flush=True)
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('UI failed to start in 60 seconds')
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
