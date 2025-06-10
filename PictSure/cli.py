import os
import click
import shutil
from .config import PRETRAINED
import pkg_resources

@click.group()
def cli():
    """PictSure command line interface."""
    pass

@cli.command()
def list_models():
    """List all available and downloaded models."""
    click.echo("Available PictSure Models:")
    click.echo("=" * 50)
    
    # Get the package directory
    package_dir = os.path.dirname(pkg_resources.resource_filename('PictSure', '__init__.py'))
    
    for model_name, model_info in PRETRAINED.items():
        # Check if model is downloaded using absolute path
        local_folder = os.path.join(package_dir, 'weights', model_info['name'])
        weights_path = os.path.join(local_folder, 'weights.pt')
        is_downloaded = os.path.exists(weights_path)
        
        # Create status indicator
        status = "✓ Downloaded" if is_downloaded else "✗ Not downloaded"
        status_color = "green" if is_downloaded else "red"
        
        # Print model information
        click.echo(f"\nModel: {click.style(model_info['name'], bold=True)}")
        click.echo(f"Status: {click.style(status, fg=status_color)}")
        click.echo(f"Type: {model_info['embed_model']}")
        click.echo(f"Resolution: {model_info['resolution']}")
        click.echo(f"Number of classes: {model_info['num_classes']}")
        click.echo(f"Transformer heads: {model_info['nheads']}")
        click.echo(f"Transformer layers: {model_info['nlayer']}")
        click.echo(f"Model size: {model_info['size']} Million Parameters")
        click.echo(f"Path: {weights_path}")
        click.echo("-" * 50)

@cli.command()
@click.argument('model_name', type=click.Choice([info['name'] for info in PRETRAINED.values()]))
@click.option('--force', '-f', is_flag=True, help='Skip confirmation prompt')
def remove(model_name, force):
    """Remove the weights of a specific model."""
    # Get the package directory
    package_dir = os.path.dirname(pkg_resources.resource_filename('PictSure', '__init__.py'))
    
    # Find the model info by name
    model_info = next((info for info in PRETRAINED.values() if info['name'] == model_name), None)
    if not model_info:
        click.echo(click.style(f"Model {model_name} not found.", fg='red'))
        return
    
    # Construct paths
    local_folder = os.path.join(package_dir, 'weights', model_info['name'])
    weights_path = os.path.join(local_folder, 'weights.pt')
    
    if not os.path.exists(weights_path):
        click.echo(click.style(f"Model {model_info['name']} is not downloaded.", fg='yellow'))
        return
    
    if not force:
        if not click.confirm(f"Are you sure you want to remove the weights for {click.style(model_info['name'], bold=True)}?"):
            click.echo("Operation cancelled.")
            return
    
    try:
        # Remove the weights file
        os.remove(weights_path)
        # Try to remove the directory if it's empty
        try:
            os.rmdir(local_folder)
        except OSError:
            pass  # Directory might not be empty, which is fine
        
        click.echo(click.style(f"Successfully removed weights for {model_info['name']}.", fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error removing weights: {str(e)}", fg='red'))

if __name__ == '__main__':
    cli() 