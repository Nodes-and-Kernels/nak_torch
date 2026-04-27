# %%
import nest_asyncio
import os

nest_asyncio.apply() # See pystan documentation on why you need this when doing jupyter
from posteriordb import PosteriorDatabaseGithub  # noqa: E402

# %%
if "GITHUB_PAT" not in os.environ.keys():
    raise ValueError("Expected GITHUB_PAT to be in environment. Please add this into, e.g., your .env file.")

my_pdb = PosteriorDatabaseGithub()
pos = my_pdb.posterior_names()

# %%
posterior = my_pdb.posterior("eight_schools-eight_schools_centered")

# %%
