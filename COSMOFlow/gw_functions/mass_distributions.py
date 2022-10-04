def PL_m1m2(Nsamples, alpha, Mmax = 100, Mmin = 5):
    def draw_cumulative(N,alpha, distribution):
        #grid = np.linspace(-23,-5,100)
        cdf = np.zeros(len(m_vec))
        for i in range(len(m_vec)):
            cdf[i] = quad(lambda M: distribution(alpha, M),  m_vec  [0], m_vec  [i])[0]
        cdf = cdf/np.max(cdf)     
        t = rn.random(N)
        samples = np.interp(t,cdf,m_vec )
        return samples

    def PL_mass1(alpha,m1):
        return m1**(-alpha)

    Norm = quad(lambda m: PL_mass1(alpha, m), Mmin, Mmax)[0]

    m_vec = np.linspace(Mmin ,Mmax, 25000)

    def m2_sample_m1(m1):
        return np.random.uniform(Mmin, m1, size = 1)[0]

    m1 = draw_cumulative(Nsamples,alpha, PL_mass1)


    m2_list = []     
    for m1_sample in m1:

        m2_list.append(m2_sample_m1(m1_sample))


    m2 = np.array(m2_list)
    return m1, m2


def PL_PEAK(Nsamples, parameters, Mmax = 100, Mmin = 5):
