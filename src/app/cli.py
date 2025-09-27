import click
from app.config import settings

@click.group()
def cli():
    """Multi-agent trading CLI."""
    pass

@cli.command()
@click.option("--amount", type=float, required=True, help="Investment amount in ₹")
def run(amount):
    """Run the main DAG pipeline (paper mode by default)."""
    click.echo(f"Starting pipeline for ₹{amount} in {settings.app_env} mode...")
    # Here you will eventually call orchestration.graph_pipeline.run_pipeline(amount)
    click.echo("Pipeline run complete (placeholder).")

@cli.command()
def status():
    """Show pipeline or system status."""
    click.echo("System status: OK (placeholder)")

if __name__ == "__main__":
    cli()
