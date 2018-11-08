using DelimitedFiles, Random
Random.seed!(92906)

data = readdlm("HMP_SRA_gut/HMP_SRA_gut_tiny_meta.tsv", '\t')

three_cat_strcol = vcat("ENV4", rand(["A", "B", "C"], size(data, 1)-1))
three_cat_intcol = vcat("ENV5", rand([1, 2, 3], size(data, 1)-1))

data_three_cat = hcat(data, three_cat_strcol, three_cat_intcol)

writedlm("HMP_SRA_gut/HMP_SRA_gut_tiny_meta_oneHotTest.tsv", data_three_cat, '\t')
