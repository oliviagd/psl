import numpy as np

THRESHOLD = 1e-6

def Estep(data, G, prob, means, Sigma):
    """
    E-step: compute the responsibility matrix

    Parameters:
        data (ndarray)
        G (int)
        prob (ndarray): mixing weights for each Gaussian component
        means (ndarray): means of Gaussian components
        Sigma (ndarray): cov matrix
    """
    # compute Gaussian PDF values for all data points and components
    n, p = data.shape
    
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_factor = 1.0 / np.sqrt((2 * np.pi) ** p * det_Sigma)
    
    diff = data[:, np.newaxis, :] - means.T
    
    exponent = -0.5 * np.einsum('nik,kl,nil->ni', diff, Sigma_inv, diff)    
    pdf_matrix = norm_factor * np.exp(exponent)
    
    # multiply by mixing weights
    weighted_pdfs = pdf_matrix * prob
    
    # normalize to get responsibilities
    resp = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
    
    return resp

def Mstep(data, resp):
    """
    M-step: update parameters based on responsibilities

    Parameters:
        data (ndarray)
        resp (ndarray): responsibility matrix from E-step, shape (n, G)
    """
    n, p = data.shape
    G = resp.shape[1]

    Nk = resp.sum(axis=0)  # sum of responsibilities for each component
    prob = Nk / n

    means = np.dot(data.T, resp) / Nk

    Sigma = np.zeros((p, p))
    for k in range(G):
        diff = data - means[:, k].T 
        Sigma += np.dot((resp[:, k][:, np.newaxis] * diff).T, diff)
    Sigma /= n

    return prob, means, Sigma

def loglik(data, G, prob, means, Sigma):
    """
    Compute the log-likelihood of the data given the current parameters of the Gaussian mixture model.
    Parameters:
        data (ndarray)
        G (int)
        prob (ndarray): mixing weights for each Gaussian component
        means (ndarray): means of Gaussian components
        Sigma (ndarray): cov matrix   
    """
    n, p = data.shape
    '''
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    normalization_const = 1 / ((2 * np.pi) ** (p / 2) * np.sqrt(Sigma_det))
    
    # data minus means for each component
    diff = data[:, np.newaxis, :] - means.T
    
    exponent = -0.5 * np.einsum('nkp, pq, nkq -> nk', diff, Sigma_inv, diff)
    
    # gaussian densities: (n, G)
    component_densities = normalization_const * np.exp(exponent)
    '''
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_factor = 1.0 / np.sqrt((2 * np.pi) ** p * det_Sigma)
    
    diff = data[:, np.newaxis, :] - means.T
    
    exponent = -0.5 * np.einsum('nik,kl,nil->ni', diff, Sigma_inv, diff)    
    pdf_matrix = norm_factor * np.exp(exponent)
    weighted_densities = pdf_matrix * prob 
    total_density = np.sum(weighted_densities, axis=1)
    log_likelihood = np.sum(np.log(total_density))
    
    return log_likelihood

def myEM(data, G, prob, means, Sigma, itmax=100):
    """
    Runner
    
    Parameters:
        data (ndarray): data, shape (n, p)
        G (int): number of Gaussian components
        prob (ndarray): initial probability vector
        means (ndarray): initial means
        Sigma (ndarray): initial cov matrix
        itmax (int): maximum number of iterations

    Returns:
        prob (ndarray): final probability vector
        means (ndarray): final means for each Gaussian component
        Sigma (ndarray): final shared cov matrix
        loglik (float): final log-likelihood of the model
    """

    log_likelihoods = []

    for iteration in range(itmax):
        resp = Estep(data, G, prob, means, Sigma)

        prob, means, Sigma = Mstep(data, resp)

        current_loglik = loglik(data, G, prob, means, Sigma)
        log_likelihoods.append(current_loglik)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < THRESHOLD:
            break

    return prob, means, Sigma, log_likelihoods[-1]
