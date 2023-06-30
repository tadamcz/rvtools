import pytest
from mkcodes import github_codeblocks

from rvtools import PROJECT_ROOT


@pytest.fixture(params=github_codeblocks(PROJECT_ROOT / "README.md", safe=False)["py"])
def block(request):
    return request.param


def test(block):
    exec(block)
