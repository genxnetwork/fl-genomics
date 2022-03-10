import numpy
from sklearn.preprocessing import StandardScaler


def ri(high):
        return numpy.random.randint(high)


class InverseEffectsGenerator:
    def __init__(self, linear_count=20, noise_size=0.2, linear_size=0.4, epistasis_size=0.4, epistasis_count=10):
        """Generates nonlinear phenotype with linear effects inversely proportional to MAF

        Example:

        .. testcode::
        from sklearn.metrics import r2_score
        from utils.simulator import InverseEffectsGenerator

        generator = InverseEffectsGenerator()
        X = numpy.random.randint(0, high=3, size=(100, 100))
        linear_y, epi_y, noise = generator.fit_transform(X)
        y = linear_y + epi_y + noise
        # should be roughly equal to linear_size + noise_size parameters of the generator
        perfect_r2 = r2_score(linear_y + epi_y, y) 

        Args:
            linear_count (int, optional): Number of significant SNPs with linear effect. Defaults to 20.
            noise_size (float, optional): Noise variance. Defaults to 0.2.
            linear_size (float, optional): Total linear phenotype variance. Defaults to 0.4.
            epistasis_size (float, optional): Total pairwize epistatic effects variance. 
                                              Set to 0.0 if phenotype should be linear. Defaults to 0.4. 
            epistasis_count (int, optional): Count of pairs of SNPs in epistasis. Defaults to 10.
        """        
        self.linear_count = linear_count
        self.epistasis_count = epistasis_count
        
        self.noise_size = noise_size
        self.linear_size = linear_size
        self.epistasis_size = epistasis_size
        
        self.scale = 1.0
    

    def linear_share(self, X: numpy.ndarray):
        """Generates linear phenotype using {self.linear_count} SNPs from genotype matrix X

        Args:
            X (numpy.ndarray): Genotype matrix with values of SNPs from {0,1,2}

        Returns:
            Tuple: linear phenotype, linear effects vector and significant SNPs mask 
        """        
        mask = numpy.random.choice(numpy.arange(X.shape[1]), size=self.linear_count, replace=False)
        X_sel = X[:, mask]
        maf = 0.5*X_sel.mean(axis=0)
        maf[maf < 1.0/X_sel.shape[0]] = 1.0/X_sel.shape[0]
        maf_mean = maf.mean()
        effects = maf_mean*numpy.random.randn(X_sel.shape[1])*self.scale / maf
        effects[effects > 5.0] = 5.0
        effects[effects < -5.0] = -5.0
        y = X_sel.dot(effects.T)
        return y, effects, mask

    def uniform_epistasis(self, X, y):
        effect_size = self.scale

        effects = {}
        combos = {}
        # firstly, simulate epistatic effects between linearly associated snps
        add = numpy.zeros((y.shape[0]))
        for pair_index in range(self.epistasis_count // 2):
            
            snp1, snp2 = ri(self.linear_count), ri(self.linear_count)
            effect = numpy.random.randn() * effect_size
            if snp1 == snp2:
                continue

            effects[(snp1, snp2)] = effects.get((snp1, snp2), 0) + effect
            snp_value1, snp_value2 = ri(3), ri(3)
            combos[(snp1, snp2)] = (snp_value1, snp_value2)
            x0, x1 = X[:, self.mask][:, snp2], X[:, self.mask][:, snp2]
            add += effect*x0*x1*(x0 == snp_value1)*(x1 == snp_value2)
        
        # secondly, just choose at random
        for pair_index in range(self.epistasis_count - len(effects)):
            snp1, snp2 = ri(X.shape[1]), ri(X.shape[1])
            effect = numpy.random.randn() * effect_size
            if snp1 == snp2:
                continue

            effects[(snp1, snp2)] = effects.get((snp1, snp2), 0) + effect
            snp_value1, snp_value2 = ri(3), ri(3)
            combos[(snp1, snp2)] = (snp_value1, snp_value2)
            x0, x1 = X[:, snp2], X[:, snp2]
            add += effect*x0*x1*(x0 == snp_value1)*(x1 == snp_value2)

        return add, effects, combos
    
    def random_noise(self, X, y):
        return numpy.random.randn(y.shape[0])


    def fit_transform(self, X: numpy.ndarray):
        """Generates nonlinear phenotype as a tuple with scaled linear phenotype, pairwise epistatic phenotype and random noise

        Args:
            X (numpy.ndarray): Genotype matrix with values of SNPs from {0,1,2}

        Returns:
            Tuple: linear phenotype, epistatic phenotype and random noise
        """        
        self.linear_y, self.linear_effects, self.mask = self.linear_share(X)
        self.noise = self.random_noise(X, self.linear_y)
        self.unif_ep, self.ep_effects, self.ep_combos = self.uniform_epistasis(X, self.linear_y)
        scaler = StandardScaler()
        
        self.linear_y = scaler.fit_transform(self.linear_y.reshape(-1, 1)) * self.linear_size**0.5
        self.unif_ep = scaler.fit_transform(self.unif_ep.reshape(-1, 1)) * self.epistasis_size**0.5
        self.noise = scaler.fit_transform(self.noise.reshape(-1, 1)) * self.noise_size**0.5
        
        return self.linear_y.ravel(), self.unif_ep.ravel(), self.noise.ravel()
