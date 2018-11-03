import numpy as np
from scipy.spatial.distance import euclidean

# chaque item a deux attributs : categorie, poids

# objectif :
# minimiser la somme des distances panier - panier_optimal
# minimiser la somme des distances poids - poids_optimal


#     Local search
#     solution initiale
#     neighbourhood function
#     acceptance strategy
#     critere stop
#     
#     first accept vs best accept


def main():

    n_categorie = 3
    panier_optimal = np.asarray([0.3, 0.6, 0.1]) #{1: 0.3, 2: 0.6, 3: 0.1}
    poids_optimal = 300.0  # grammes
    n_items = 100
    model = LocalSearchModel(n_categorie, n_items, poids_optimal, panier_optimal)

    model.initialise()

    print('Initial score')
    score, score_poids, score_cat = model.calc_score(model.matrice_items_paniers)
    print(score, score_poids, score_cat)
    model.score = score

    # probleme de poids !!!
    cutoff = 0.1
    while True:
        model.update_panier()


class LocalSearchModel:

    def __init__(self, n_categorie, n_items, poids_optimal, panier_optimal):
        self.n_categorie = n_categorie
        self.n_items = n_items
        self.poids_optimal = poids_optimal
        self.panier_optimal = panier_optimal


    def initialise(self): 
    
        poids_max = 200.0
        # (distribution uniforme de poids)
        self.items_poids = np.random.random(size=self.n_items) * poids_max + 1.0
        # (distribution uniforme de categories)
        self.items_cat = np.random.random_integers(low=1, 
                                                   high=self.n_categorie, 
                                                   size=self.n_items)  
        
        poids_total = np.sum(self.items_poids)
       
        # estimation du nombre de paniers 
        self.n_paniers = int(np.ceil(poids_total / self.poids_optimal))
        print(self.n_paniers)
        
        self.matrice_items_paniers = np.zeros((self.n_paniers, self.n_items), dtype=np.int32)
    
        # placer chaque item dans un panier choisi de facon aleatoire
        # chaque item peut etre dans un panier seulement
        # TODO meilleure initialisation
        indices_panier = np.random.random_integers(low=0, 
                                                   high=self.n_paniers-1, 
                                                   size=self.n_items)
        self.matrice_items_paniers[indices_panier, np.arange(self.n_items)] = 1
    
      
    def calc_score(self, matrice_items_paniers):
 
        score_poids = self.calc_score_poids(matrice_items_paniers)
        score_cat = self.calc_score_cat(matrice_items_paniers)
        score = 0.01*score_poids + score_cat

        return score, score_poids, score_cat
       
 
    def update_panier(self):
   
        for i in range(self.n_items):
            matrice_items_paniers = self.matrice_items_paniers 

            item = matrice_items_paniers[:, i]
            
            current_index = np.where(item==1)[0][0]
            # transferer l'item dans un autre panier
            index = np.random.random_integers(low=0, high=self.n_paniers-1)
            if index != current_index:
                matrice_items_paniers[current_index, i] = 0
                matrice_items_paniers[index, i] = 1
                score, score_poids, score_cat = self.calc_score(matrice_items_paniers)
                if score < self.score:
                    self.matrice_items_paniers = matrice_items_paniers
                    print(score, score_poids, score_cat)
                    print(self.calc_distribution_par_panier(matrice_items_paniers))
                    print(self.calc_poids_par_panier(matrice_items_paniers))
                    self.score = score
    
    def calc_poids_par_panier(self, matrice_items_paniers):

        poids_par_panier = np.row_stack([self.items_poids] * self.n_paniers) * matrice_items_paniers
        poids_par_panier = np.sum(poids_par_panier, axis=1)
    
        assert np.isclose(poids_par_panier.sum(), self.items_poids.sum())

        return poids_par_panier

 
    def calc_score_poids(self, matrice_items_paniers):
        '''
        score = somme sur tous les paniers de (poids_panier - poids_optimal)^2
        '''
    
        poids_par_panier = self.calc_poids_par_panier(matrice_items_paniers)
 
        distance_poids = poids_par_panier - self.poids_optimal
    
        return np.sqrt(np.sum(distance_poids * distance_poids))
   

    def calc_distribution_par_panier(self, matrice_items_paniers): 
    
        # distributions des categories pour chaque panier
        cat_par_panier = np.row_stack([self.items_cat] * self.n_paniers) * matrice_items_paniers
        distribution_par_panier = np.apply_along_axis(distribution_panier, 
                                                      axis=1, 
                                                      arr=cat_par_panier, 
                                                      n_categorie=self.n_categorie)
    
        # verifier que somme des probabilites = 1
        assert np.all(np.logical_or(np.isclose(distribution_par_panier.sum(axis=1), 1.0), (distribution_par_panier.sum(axis=1) == 0.0)))

        return distribution_par_panier

    
    def calc_score_cat(self, matrice_items_paniers):

        # distributions des categories pour chaque panier
        distribution_par_panier = self.calc_distribution_par_panier(matrice_items_paniers)

        # distribution_par_panier = np.asarray([[0.32, 0.6, 0.08], [0.0, 0.0, 1.0]])
        distance_panier = np.apply_along_axis(euclidean, 
                                              axis=1, 
                                              arr=distribution_par_panier, 
                                              v=self.panier_optimal.reshape(-1, 1))
    
        return np.sum(distance_panier)
    
    
def distribution_panier(a, n_categorie):
    '''
    0 = cet item n'est pas dans le panier
    Par exemple [0, 2, 3, 1, 0, 0, 0, 0] => [0.333, 0.333, 0.333]
    '''

    a = a[a > 0]

    n_items = float(len(a))
    unique, counts = np.unique(a, return_counts=True)

    d = np.zeros(n_categorie, dtype=np.float64)
    d[unique - 1] = counts / n_items

    return d


if __name__ == '__main__':
    main()
