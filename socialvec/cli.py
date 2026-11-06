"""Console script for socialvec."""
import sys

import click

from socialvec import __version__
from socialvec.exceptions import SocialVecError

DEFAULT_MODEL = "2020"
CLASSIFIER_MODEL = "2020c"


def _load(model_name):
    # Imported here so `socialvec --help` stays fast.
    from socialvec.socialvec import SocialVec
    return SocialVec(model_name=model_name)


def _model_option(default):
    return click.option("--model", "model_name", default=default, show_default=True,
                        help="SocialVec model to use.")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="socialvec")
def main():
    """SocialVec: social embeddings of popular Twitter accounts."""


@main.command()
@click.argument("user")
@click.option("--topn", "-n", default=10, show_default=True, type=click.IntRange(1, 1000),
              help="Number of similar accounts to show.")
@_model_option(DEFAULT_MODEL)
def similar(user, topn, model_name):
    """Show the accounts most similar to USER (a user ID or username)."""
    sv = _load(model_name)
    df = sv.get_similar(user, topn)
    click.echo(df[["twitter_id", "similarity", "screen_name", "name"]].to_string(index=False))


@main.command()
@click.argument("user1")
@click.argument("user2")
@_model_option(DEFAULT_MODEL)
def similarity(user1, user2, model_name):
    """Show the cosine similarity between USER1 and USER2."""
    sv = _load(model_name)
    click.echo(f"{sv.get_similarity(user1, user2):.4f}")


@main.command()
@click.argument("classifier")
@click.argument("user")
@_model_option(CLASSIFIER_MODEL)
def predict(classifier, user, model_name):
    """Predict a trait of USER with CLASSIFIER (e.g. political, gender, age)."""
    sv = _load(model_name)
    sv.init_classifier()
    label, confidence = sv.classifier.predict(classifier, sv[user])
    click.echo(f"{label}\t(confidence {confidence:.3f})")


@main.command()
@_model_option(DEFAULT_MODEL)
def download(model_name):
    """Download a model and the account metadata ahead of time."""
    _load(model_name)
    click.echo(f"Model {model_name} is ready.")


def run():
    """Entry point: show SocialVec errors as a message instead of a traceback."""
    try:
        return main(standalone_mode=False)
    except click.exceptions.Abort:
        click.echo("Aborted!", err=True)
        return 1
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except SocialVecError as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(run())  # pragma: no cover
