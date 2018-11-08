### create encoded data

using DelimitedFiles, Random, Distributions
Random.seed!(92906)

data = readdlm("HMP_SRA_gut/HMP_SRA_gut_tiny_meta.tsv", '\t')

three_cat_strcol = vcat("ENV4", rand(["A", "B", "C"], size(data, 1)-1))
three_cat_intcol = vcat("ENV5", rand([1, 2, 3], size(data, 1)-1))
cont_var_col = vcat("ENV6", rand(Normal(), size(data, 1)-1))

data_three_cat = hcat(data, three_cat_strcol, three_cat_intcol, cont_var_col)

writedlm("HMP_SRA_gut/HMP_SRA_gut_tiny_meta_oneHotTest.tsv", data_three_cat, '\t')

#--------

using DelimitedFiles, FileIO
using Flux:onehotbatch
data, header = readdlm("HMP_SRA_gut/HMP_SRA_gut_tiny_meta_oneHotTest.tsv", '\t', header=true)
onehot_cols = []
onehot_vnames = []
for j in 1:size(data, 2)
    v = data[:, j]
    vn = header[j]
    if j != size(data, 2) && length(unique(v)) > 2
        cs_sorted = sort(unique(v))
        v_enc = Matrix{Int}(permutedims(onehotbatch(v, cs_sorted)))
        vn_enc = [vn * "_" * string(c) for c in cs_sorted]
    else
        v_enc = hcat(v)
        vn_enc = [vn]
    end
    push!(onehot_cols, v_enc)
    push!(onehot_vnames, vn_enc)
end
data_onehot = hcat(onehot_cols...)
header_onehot = vcat(onehot_vnames...)
d = load("preprocessing_expected.jld2")
d["exp_dict"]["meta_tiny_oneHotTest"] = (meta_data=data_onehot, meta_header=header_onehot)
save("preprocessing_expected.jld2", collect(Iterators.flatten(d))...)
