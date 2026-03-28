"""
CeDNe Web Server CLI

Starts the CeDNe web backend (FastAPI + Uvicorn) for the interactive
network visualization frontend.

Usage:
    cedne serve              # Start on default port 8000
    cedne serve --port 9000  # Custom port
    cedne serve --host 0.0.0.0  # Allow external connections
"""
import click
import sys
import os


def _find_backend_main():
    """Locate the cedne_web backend main.py relative to the package."""
    # When installed: main.py is in cedne_web/backend/ relative to the repo root
    # Try several possible locations
    candidates = [
        # Relative to this file (src/cedne/serve.py -> ../../cedne_web/backend)
        os.path.join(os.path.dirname(__file__), '..', '..', 'cedne_web', 'backend'),
        # Relative to CWD
        os.path.join(os.getcwd(), 'cedne_web', 'backend'),
    ]
    for path in candidates:
        main_path = os.path.join(path, 'main.py')
        if os.path.exists(main_path):
            return os.path.abspath(path)
    return None


@click.group()
def cli():
    """CeDNe - web"""
    pass


@cli.command()
@click.option('--port', default=8000, help='Port to run the server on (default: 8000)')
@click.option('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port, host, reload):
    """Start the CeDNe web visualization backend."""
    import uvicorn

    backend_dir = _find_backend_main()
    if backend_dir is None:
        click.echo(click.style("Error: ", fg='red') +
                    "Could not find cedne_web/backend/main.py. "
                    "Make sure you're running from the CeDNe repository root, "
                    "or that the package is properly installed.")
        sys.exit(1)

    # Add the backend directory to the Python path so uvicorn can find 'main'
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # Banner
    click.echo()
    click.echo(click.style("  ╔═══════════════════════════════════╗", fg='cyan'))
    click.echo(click.style("  ║", fg='cyan') +
               click.style("   CeDNe Web Visualization Server  ", fg='white', bold=True) +
               click.style("║", fg='cyan'))
    click.echo(click.style("  ╚═══════════════════════════════════╝", fg='cyan'))
    click.echo()
    click.echo(f"  Backend:  {click.style(f'http://{host}:{port}', fg='green', bold=True)}")
    click.echo(f"  Frontend: {click.style('http://localhost:5173', fg='yellow')} (start separately with npm run dev)")
    click.echo()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
