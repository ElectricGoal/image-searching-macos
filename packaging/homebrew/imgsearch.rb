# Homebrew formula template for imgsearch.
#
# To use this in a personal tap:
#   1. Create a repo named `homebrew-imgsearch` under your GitHub account.
#   2. Copy this file to Formula/imgsearch.rb in that repo.
#   3. Update `url` and `sha256` to the published PyPI sdist.
#   4. Regenerate the `resource` blocks with:
#        homebrew-pypi-poet -f imgsearch
#      (run inside a fresh venv that has ONLY `imgsearch` installed).
#   5. `brew tap <you>/imgsearch && brew install imgsearch`.
#
# Note: faiss-cpu ships arm64 wheels on PyPI. This formula relies on
# `virtualenv_install_with_resources` picking the wheel where available.

class Imgsearch < Formula
  include Language::Python::Virtualenv

  desc "Semantic image search CLI with automatic backend selection"
  homepage "https://github.com/toxu/imgsearch"
  url "https://files.pythonhosted.org/packages/source/i/imgsearch/imgsearch-0.1.0.tar.gz"
  sha256 "REPLACE_WITH_SDIST_SHA256"
  license "MIT"

  depends_on "python@3.12"
  depends_on macos: :ventura
  depends_on arch: :arm64

  # resource blocks go here — generate with homebrew-pypi-poet

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"imgsearch", "--version"
  end
end
