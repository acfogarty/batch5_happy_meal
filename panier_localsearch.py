import numpy as np

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
    panier_optimal = {0: 0.3, 1: 0.6, 2: 0.1}
    poids_optimal = 300.0  # grammes
    n_items = 10

    items_poids, items_cat, matrice_items_panier, n_paniers = initialise(n_categorie, 
                                                                         n_items, 
                                                                         poids_optimal)

    score = \
       calc_score_poids(items_poids, poids_optimal, matrice_items_panier, n_paniers) + \
       calc_score_cat(items_cat, panier_optimal, matrice_items_panier, n_paniers)
    print(score)


def initialise(n_categorie, n_items, poids_optimal): 

    poids_max = 200.0
    # (distribution uniforme de poids)
    items_poids = np.random.random(size=n_items) * poids_max + 1.0
    # (distribution uniforme de categories)
    items_cat = np.random.random_integers(low=0, high=n_categorie-1, size=n_items)  
    
    poids_total = np.sum(items_poids)
   
    # estimation du nombre de paniers 
    n_paniers = int(np.ceil(poids_total / poids_optimal))
    
    matrice_items_paniers = np.zeros((n_items, n_paniers), dtype=np.int32)

    # placer chaque item dans un panier choisi de facon aleatoire
    # chaque item peut etre dans un panier seulement
    # TODO meilleure initialisation
    indices_panier = np.random.random_integers(low=0, high=n_paniers-1, size=n_items)
    matrice_items_paniers[np.arange(n_items), indices_panier] = 1

    return items_poids, items_cat, matrice_items_paniers, n_paniers
   
 
def calc_score_poids(items_poids, poids_optimal, matrice_items_panier, n_paniers):

    # poids par panier
    poids_par_panier = np.column_stack([items_poids] * n_paniers) * matrice_items_panier
    poids_par_panier = np.sum(poids_par_panier, axis=0)

    # assert poids_par_panier.sum() == items_poids.sum()
    
    distance_poids = poids_par_panier - poids_optimal

    return np.sum(distance_poids * distance_poids)


def calc_score_cat(items_cat, panier_optimal, matrice_items_panier, n_paniers):
    print(matrice_items_panier)
    return 0
    #distance_panier = 


if __name__ == '__main__':
    main()
