from scipy.stats import rv_continuous


class Distribution(rv_continuous):

    def __init__(self, pdf_fun):
        super().__init__()
        self.pdf_fun = pdf_fun

    def _pdf(self, x, *args):
        return self.pdf_fun(x)

