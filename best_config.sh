bash run.sh train ConvE FB15k-237 0 1 128 256 200 9.0 1.0 0.000002 100000 16
bash run.sh train ConvE FB15k-237 0 2 128 256 200 9.0 1.0 0.000005 100000 16
bash run.sh train ConvE FB15k-237 0 3 128 256 200 9.0 1.0 0.00001 100000 16
bash run.sh train ConvE FB15k-237 1 4 128 256 200 9.0 1.0 0.00002 100000 16
bash run.sh train ConvE FB15k-237 1 5 128 256 200 9.0 1.0 0.00005 100000 16
bash run.sh train ConvE FB15k-237 1 6 128 256 200 9.0 1.0 0.0001 100000 16

bash run.sh train ConvE FB15k-237 0 10 128 256 200 9.0 1.0 0.0001 100000 16

bash run.sh train ConvE FB15k-237 0 12 128 256 200 9.0 1.0 0.003 10000 128


bash run.sh train CoCo FB15k-237 0 13 128 256 200 9.0 1.0 0.003 10000 128
bash run.sh train CoCo FB15k-237 0 14 128 256 200 9.0 1.0 0.001 10000 128
bash run.sh train CoCo FB15k-237 0 15 128 256 200 9.0 1.0 0.002 10000 128
bash run.sh train CoCo FB15k-237 0 16 128 256 200 9.0 1.0 0.004 10000 128
bash run.sh train CoCo FB15k-237 0 17 128 256 200 9.0 1.0 0.001 20000 128

bash run.sh train ConvE FB15k-237 0 18 128 256 200 9.0 1.0 0.003 20000 128

bash run.sh train CoCo2 FB15k-237 0 19 128 256 200 9.0 1.0 0.001 40000 128

bash run.sh train ConvE FB15k-237 0 20 128 256 200 9.0 1.0 0.003 40000 128


# TransE
bash run.sh train TransE FB15k 0 22 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k-237 1 21 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 23 512 1024 500 12.0 0.5 0.0001 80000 8
# ComplEx
bash run.sh train ComplEx FB15k 0 24 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002     #
bash run.sh train ComplEx FB15k-237 0 25 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18 1 26 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
# ConvE
bash run.sh train ConvE FB15k 1 27 128 256 200 9.0 1.0 0.003 40000 128
bash run.sh train ConvE FB15k-237 0 20 128 256 200 9.0 1.0 0.003 40000 128
bash run.sh train ConvE wn18 1 28 128 256 200 9.0 1.0 0.003 40000 128


# CoCo2
bash run.sh train CoCo2 FB15k 0 29 1024 256 200 9.0 1.0 0.003 40000 16
bash run.sh train CoCo2 FB15k 1 30 1024 256 200 9.0 1.0 0.001 40000 16
bash run.sh train CoCo2 FB15k 0 31 1024 256 200 9.0 1.0 0.0005 40000 16
bash run.sh train CoCo2 FB15k 1 32 1024 256 200 9.0 1.0 0.0002 40000 16


bash run.sh train CoCo2 FB15k-237 0 33 1024 256 200 9.0 1.0 0.003 40000 16    # without 'b'   not as good as CoCo or CoCo2 with 'b'
bash run.sh train CoCo2 FB15k-237 0 34 1024 256 200 9.0 1.0 0.001 40000 16
bash run.sh train CoCo2 FB15k-237 1 35 1024 256 200 9.0 1.0 0.0005 40000 16
bash run.sh train CoCo2 FB15k-237 1 36 1024 256 200 9.0 1.0 0.0002 40000 16


bash run.sh train CoCo2 wn18 1 37 1024 256 200 9.0 1.0 0.003 40000 16       # without 'b' extremely bad
bash run.sh train CoCo2 wn18 1 38 1024 256 200 9.0 1.0 0.001 40000 16
bash run.sh train CoCo2 wn18 0 39 1024 256 200 9.0 1.0 0.0005 40000 16
bash run.sh train CoCo2 wn18 0 40 1024 256 200 9.0 1.0 0.0002 40000 16


#######################################################################################################################
#######################################################################################################################

# tmux new -s #name
# tmux a -t #name
# tmux ls
# tmux kill-session -t myname

bash run.sh train CoCo2 FB15k-237 0 43 1024 256 200 9.0 1.0 0.003 40000 16    # with 'b'    # (10 15 24)
bash run.sh train CoCo2 FB15k-237 0 44 1024 256 200 9.0 1.0 0.001 40000 16                  # (8, 12, 19)
bash run.sh train CoCo2 FB15k-237 0 45 1024 256 200 9.0 1.0 0.0005 40000 16                 # (14, 19, 27)
bash run.sh train CoCo2 FB15k-237 0 46 1024 256 200 9.0 1.0 0.0002 40000 16                 # (11, 13, 15)



bash run.sh train CoCo2 FB15k-237 0 60 128 256 200 9.0 1.0 0.005 40000 128                  # (11, 15, 22)

bash run.sh train CoCo2 FB15k-237 0 47 128 256 200 9.0 1.0 0.003 40000 128    # with 'b'    # (15, 21, 31)
bash run.sh train CoCo2 FB15k-237 0 48 128 256 200 9.0 1.0 0.001 40000 128                  # (13, 18, 25)
bash run.sh train CoCo2 FB15k-237 0 49 128 256 200 9.0 1.0 0.0005 40000 128                 # (12, 14, 17)
bash run.sh train CoCo2 FB15k-237 0 50 128 256 200 9.0 1.0 0.0002 40000 128                 # (6, 10, 14)

#######################################################################################################################
#  New idea
# ComplEx  vs ComplExC
bash run.sh train ComplEx FB15k 1 51 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002      # (70, 81, 88)
bash run.sh train ComplEx FB15k-237 1 52 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001   # (23, 35, 51)
bash run.sh train ComplEx wn18 1 53 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001           # (85, 92, 95)

bash run.sh train ComplExC FB15k 1 54 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002     # (71, 81, 88)
bash run.sh train ComplExC FB15k-237 1 55 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001  # (22, 34, 49)
bash run.sh train ComplExC wn18 0 56 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001          # (59, 88, 94)

#######################################################################################################################

## ConvE vs ConvE2   (ConvE2 is very bad)
#bash run.sh train ConvE FB15k-237 0 57 128 256 200 9.0 1.0 0.003 40000 128                # (18, 28, 44)
#bash run.sh train ConvE2 FB15k-237 0 58 128 256 200 9.0 1.0 0.003 100000 128              # (6, 11, 17)
#bash run.sh train ConvE2 FB15k-237 0 59 128 256 200 9.0 1.0 0.001 100000 128              # (2, 5, 12)
#
#bash run.sh train ConvE2 FB15k-237 0 61 128 256 200 9.0 1.0 0.005 40000 128               # (2, 5, 12)


#######################################################################################################################
# ComplExD (running 62, 64, others haven't)
bash run.sh train ComplExD FB15k 1 62 1024 256 1000 500.0 1.0 0.003 150000 16 -de -dr -r 0.000002
bash run.sh train ComplExD FB15k 1 63 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train ComplExD FB15k 0 64 1024 256 1000 500.0 1.0 0.0005 150000 16 -de -dr -r 0.000002

bash run.sh train ComplExD FB15k 0 65 1024 256 1000 500.0 1.0 0.01 150000 16 -de -dr -r 0.000002



bash run.sh train ComplExD FB15k-237 1 67 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplExD wn18 1 68 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001


#######################################################################################################################
# DistMult vs DistMultC
bash run.sh train DistMult FB15k-237 0 71 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001    # (22, 33, 48)
bash run.sh train DistMultC FB15k-237 0 72 1024 256 2000 200.0 1.0 0.0005 100000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 0 73 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001   # not run
bash run.sh train DistMultC FB15k-237 0 74 1024 256 2000 200.0 1.0 0.002 100000 16 -r 0.00001   # not run

bash run.sh train DistMultC FB15k-237 0 75 1024 256 400 200.0 1.0 0.01 20000 16 -r 0.00001      # (13, 24, 38)
bash run.sh train DistMultC FB15k-237 0 76 256 256 400 200.0 1.0 0.004 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 0 77 256 256 400 200.0 1.0 0.002 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 0 78 256 256 400 200.0 1.0 0.001 40000 16 -r 0.00001

bash run.sh train DistMultC FB15k-237 1 79 1024 256 400 200.0 1.0 0.05 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 80 1024 256 400 200.0 1.0 0.1 40000 16 -r 0.00001

bash run.sh train DistMultC FB15k-237 1 83 1024 256 400 200.0 1.0 0.4 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 84 1024 256 400 200.0 1.0 0.2 80000 16 -r 0.00001

bash run.sh train DistMultC FB15k-237 1 85 1024 256 400 200.0 1.0 0.05 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 86 1024 256 400 200.0 1.0 0.08 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 87 1024 256 400 200.0 1.0 0.1 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 88 1024 256 400 200.0 1.0 0.12 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 89 1024 256 400 200.0 1.0 0.15 80000 16 -r 0.00001

bash run.sh train DistMultC FB15k-237 1 90 1024 256 1000 200.0 1.0 0.05 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 91 1024 256 1000 200.0 1.0 0.08 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 92 1024 256 1000 200.0 1.0 0.1 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 93 1024 256 1000 200.0 1.0 0.12 80000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 94 1024 256 1000 200.0 1.0 0.15 80000 16 -r 0.00001

bash run.sh train DistMultC FB15k-237 1 95 1024 256 100 200.0 1.0 0.1 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 96 1024 256 200 200.0 1.0 0.1 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 97 1024 256 600 200.0 1.0 0.1 40000 16 -r 0.00001
bash run.sh train DistMultC FB15k-237 1 98 1024 256 800 200.0 1.0 0.1 40000 16 -r 0.00001




#######################################################################################################################
# ComplExH (play with all 8 different combinations of Hermitian product Variant <head, relation, tail> (Re & Img)

bash run.sh train ComplExH FB15k-237 1 81 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplExH FB15k-237 0 82 1024 256 500 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001





#######################################################################################################################
bash run.sh train ComplEx FB15k-237 0 101 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train QuarterNion FB15k-237 0 102 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001

bash run.sh train QuarterNion FB15k-237 0 103 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr


bash run.sh train QuarterNion FB15k-237 0 104 1024 256 1000 200.0 1.0 0.0002 100000 16 -de -dr -r 0.00001
bash run.sh train QuarterNion FB15k-237 0 105 1024 256 1000 200.0 1.0 0.0005 100000 16 -de -dr -r 0.00001


bash run.sh train QuarterNion FB15k-237 1 106 1024 256 1000 200.0 1.0 0.0008 100000 16 -de -dr -r 0.00001
bash run.sh train QuarterNion FB15k-237 1 107 1024 256 1000 200.0 1.0 0.002 100000 16 -de -dr -r 0.00001
bash run.sh train QuarterNion FB15k-237 1 108 1024 256 1000 200.0 1.0 0.005 100000 16 -de -dr -r 0.00001
bash run.sh train QuarterNion FB15k-237 0 109 1024 256 1000 200.0 1.0 0.008 100000 16 -de -dr -r 0.00001




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# bash run.sh train RotatE FB15k    0       0      1024        256               1000         24.0    1.0   0.0001 150000         16               -de
#               1     2      3       4      5        6          7                   8          9       10     11     12           13
#              mode model  dataset  GPU  saveid    batchsize   neg_sample_size  hidden_dim    gamma   alpha   lr    Max_steps  test_batchsize


bash run.sh train ConvE FB15k-237 1 0 128 256 200 9.0 1.0 0.00002 100000 16

# Best Configuration for RotatE
#
bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de                  #
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries 
bash run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
#
# Best Configuration for pRotatE
#
bash run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for TransE
# 
bash run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16                    #
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train TransE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for ComplEx
# 
bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002     #
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#
# Best Configuration for DistMult
# 
bash run.sh train DistMult FB15k 0 0 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002
bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult wn18 0 0 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001
bash run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005
bash run.sh train DistMult countries_S1 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash run.sh train DistMult countries_S2 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash run.sh train DistMult countries_S3 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
#
