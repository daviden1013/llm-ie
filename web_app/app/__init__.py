import os
import logging
import datetime
from flask import Flask
from flask_cors import CORS # For Cross-Origin Resource Sharing

# --- Application Factory Function ---
def create_app(test_config=None):
    """
    Application factory function. Creates and configures the Flask app.
    """
    # Create the Flask app instance.
    # __name__ is the name of the current Python module.
    # instance_relative_config=True means config files are relative to the instance folder.
    # The instance folder is located outside the app package and can hold local data
    # that shouldn't be committed to version control, like secret keys or database files.
    app = Flask(__name__,
                instance_relative_config=True,
                template_folder='templates',  # Explicitly point to templates within the app package
                static_folder='static'      # Explicitly point to static within the app package
                )

    # --- Configuration ---
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_secret_key'), # Use environment variable or a default
        # Add other default configurations here
        CURRENT_TIME=datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # For download filenames
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        # This can be used for production secrets, etc.
        # e.g., app.config.from_pyfile('config.py', silent=True)
        # For now, we'll rely on environment variables or defaults.
        pass
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # --- Logging ---
    # Set up basic logging if not already configured by a parent logger
    if not app.debug: # More robust logging for production
        # Example: Log to a file or a logging service
        # For now, basic console logging is fine, Flask's default logger will be used.
        pass
    # In debug mode, Flask's default logger is usually sufficient.
    # You can customize further if needed.
    # Example: app.logger.setLevel(logging.INFO)

    # --- CORS (Cross-Origin Resource Sharing) ---
    # This is important if your frontend is served from a different domain/port
    # than your backend API, especially during development.
    # For production, you might want to restrict origins.
    CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow all origins for API routes

    # --- Register Blueprints ---
    # Import and register the main blueprint from routes.py
    from .routes import main_bp  # .routes means from routes.py in the current package (app)
    app.register_blueprint(main_bp)
    # If you had other blueprints (e.g., for auth), you would register them here too.
    # from . import auth
    # app.register_blueprint(auth.bp)

    # --- Optional: Initialize Extensions ---
    # Example: If you were using Flask-SQLAlchemy
    # from .models import db
    # db.init_app(app)

    # --- Optional: Create a simple route for health check (can also be in a blueprint) ---
    @app.route('/health')
    def health_check():
        return 'OK', 200

    app.logger.info("Flask application created and configured.")
    return app

