import os
from app import create_app

application = create_app()

if __name__ == '__main__':
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 't']

    application.logger.info(f"Starting Flask development server on {host}:{port} with debug_mode={debug_mode}")
    application.run(host=host, port=port, debug=debug_mode)

