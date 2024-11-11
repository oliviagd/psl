import numpy as np
from scipy.stats import multivariate_normal

def gaussian_pdf_matrix(data, means, Sigma):
    """
    Vectorized computation of the Gaussian PDF for all data points and components.

    Parameters:
        data (ndarray): Data points, shape (n, p)
        means (ndarray): Means of Gaussian components, shape (p, G)
        Sigma (ndarray): Covariance matrix shared by all components, shape (p, p)
        
    Returns:
        pdf_matrix (ndarray): PDF values for each data point and component, shape (n, G)
    """
    n, p = data.shape
    G = means.shape[1]
    
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_factor = 1.0 / np.sqrt((2 * np.pi) ** p * det_Sigma)
    
    # Expand data and means for broadcasting
    diff = data[:, np.newaxis, :] - means.T  # Shape (n, G, p)
    
    # Compute the exponent part in a vectorized way
    exponent = -0.5 * np.einsum('nik,kl,nil->ni', diff, Sigma_inv, diff)
    
    # Apply normalization factor
    pdf_matrix = norm_factor * np.exp(exponent)  # Shape (n, G)
    
    return pdf_matrix

def Estep(data, G, prob, means, Sigma):
    """
    E-step: Compute the responsibility matrix (posterior probabilities) in a vectorized way.

    Parameters:
        data (ndarray): Data points, shape (n, p)
        G (int): Number of Gaussian components
        prob (ndarray): Mixing weights for each Gaussian component, shape (G,)
        means (ndarray): Means of Gaussian components, shape (p, G)
        Sigma (ndarray): Covariance matrix shared by all components, shape (p, p)
        
    Returns:
        resp (ndarray): Responsibility matrix, shape (n, G)
    """
    # Step 1: Compute Gaussian PDF values for all data points and components
    pdf_matrix = gaussian_pdf_matrix(data, means, Sigma)  # Shape (n, G)
    
    # Step 2: Multiply by mixing weights
    weighted_pdfs = pdf_matrix * prob  # Broadcasting, shape (n, G)
    
    # Step 3: Normalize to get responsibilities
    resp = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)  # Shape (n, G)
    
    return resp

def Mstep(data, resp):
    """
    M-step: Update parameters based on responsibilities

    Parameters:
        data (ndarray): Data points, shape (n, p)
        resp (ndarray): Responsibility matrix from E-step, shape (n, G)

    Returns:
        prob (ndarray): Updated probabilities for each Gaussian component, shape (G,)
        means (ndarray): Updated means for each Gaussian component, shape (p, G)
        Sigma (ndarray): Updated shared covariance matrix, shape (p, p)
    """
    n, p = data.shape
    G = resp.shape[1]

    Nk = resp.sum(axis=0)  # sum of responsibilities for each component
    prob = Nk / n

    means = np.dot(data.T, resp) / Nk  # Shape (p, G)

    Sigma = np.zeros((p, p))
    for k in range(G):
        diff = data - means[:, k].T  # Shape (n, p)
        Sigma += np.dot((resp[:, k][:, np.newaxis] * diff).T, diff)
    Sigma /= n

    return prob, means, Sigma

def loglik(data, G, prob, means, Sigma):
    """
    Calculate the log-likelihood of the data given current parameters

    Parameters:
        data (ndarray): Data points, shape (n, p)
        G (int): Number of Gaussian components
        prob (ndarray): Probabilities of each component, shape (G,)
        means (ndarray): Means of Gaussian components, shape (p, G)
        Sigma (ndarray): Covariance matrix shared by all components, shape (p, p)

    Returns:
        log_likelihood (float): Log-likelihood of the data
    """
    n, p = data.shape
    log_likelihood = 0

    for i in range(n):
        likelihood_i = 0
        for k in range(G):
            likelihood_i += prob[k] * multivariate_normal.pdf(data[i], mean=means[:, k], cov=Sigma)
        log_likelihood += np.log(likelihood_i)

    return log_likelihood

def myEM(data, G, prob, means, Sigma, itmax=100):
    """
    Main EM algorithm to estimate parameters of Gaussian Mixture Model

    Parameters:
        data (ndarray): Data points, shape (n, p)
        G (int): Number of Gaussian components
        prob (ndarray): Initial probability vector, shape (G,)
        means (ndarray): Initial means, shape (p, G)
        Sigma (ndarray): Initial shared covariance matrix, shape (p, p)
        itmax (int): Maximum number of iterations

    Returns:
        prob (ndarray): Final probability vector, shape (G,)
        means (ndarray): Final means for each Gaussian component, shape (p, G)
        Sigma (ndarray): Final shared covariance matrix, shape (p, p)
        loglik (float): Final log-likelihood of the model
    """

    log_likelihoods = []

    for iteration in range(itmax):
        resp = Estep(data, G, prob, means, Sigma)

        prob, means, Sigma = Mstep(data, resp)

        current_loglik = loglik(data, G, prob, means, Sigma)
        log_likelihoods.append(current_loglik)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
            break

    return prob, means, Sigma, log_likelihoods[-1]
