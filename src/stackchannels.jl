# modified from standard queue Channel definition provided by Julia

import Base: put!, take!, push!, fetch, shift!, show, isready, wait, eltype, close, next

const DEF_CHANNEL_SZ=32

mutable struct StackChannel{T} <: AbstractChannel where T
    cond_take::Condition    # waiting for data to become available
    cond_put::Condition     # waiting for a writeable slot
    state::Symbol

    data::Array{T,1}
    sz_max::Int             # maximum size of channel

    function StackChannel{T}(sz) where T
        sz_max = sz == typemax(Int) ? typemax(Int) - 1 : sz
        new(Condition(), Condition(), :open, Array{T}(0), sz_max)
    end
end

StackChannel(sz::Int = DEF_CHANNEL_SZ) = StackChannel{Any}(sz)

closed_exception() = InvalidStateException("StackChannel is closed.", :closed)

"""
    close(c::StackChannel)
Closes a channel. An exception is thrown by:
* `put!` on a closed channel.
* `take!` and `fetch` on an empty, closed channel.
"""
function close(c::StackChannel)
    c.state = :closed
    notify_error(c::StackChannel, closed_exception())
    nothing
end
isopen(c::StackChannel) = (c.state == :open)

mutable struct InvalidStateException <: Exception
    msg::AbstractString
    state::Symbol
end

"""
    put!(c::StackChannel, v)
Appends an item `v` to the channel `c`. Blocks if the channel is full.
"""
function put!(c::StackChannel, v)
    !isopen(c) && throw(closed_exception())
    while length(c.data) == c.sz_max
        wait(c.cond_put)
    end
    push!(c.data, v)
    notify(c.cond_take, nothing, true, false)  # notify all, since some of the waiters may be on a "fetch" call.
    v
end

push!(c::StackChannel, v) = put!(c, v)

function fetch(c::StackChannel)
    wait(c)
    c.data[1]
end

"""
    take!(c::StackChannel)
Removes and returns a value from a `StackChannel`. Blocks till data is available.
"""
function take!(c::StackChannel)
    wait(c)
    v = pop!(c.data)
    notify(c.cond_put, nothing, false, false) # notify only one, since only one slot has become available for a put!.
    v
end

shift!(c::StackChannel) = take!(c)

"""
    isready(c::StackChannel)
Determine whether a `StackChannel` has a value stored to it.
`isready` on `StackChannel`s is non-blocking.
"""
isready(c::StackChannel) = n_avail(c) > 0

function wait(c::StackChannel)
    while !isready(c)
        !isopen(c) && throw(closed_exception())
        wait(c.cond_take)
    end
    nothing
end

function notify_error(c::StackChannel, err)
    notify_error(c.cond_take, err)
    notify_error(c.cond_put, err)
end

eltype(::Type{StackChannel{T}}) where {T} = T

n_avail(c::StackChannel) = length(c.data)

show(io::IO, c::StackChannel) = print(io, "$(typeof(c))(sz_max:$(c.sz_max),sz_curr:$(n_avail(c)))")

start(c::StackChannel{T}) where {T} = Ref{Nullable{T}}()
function done(c::StackChannel, state::Ref)
    try
        # we are waiting either for more data or channel to be closed
        state[] = take!(c)
        return false
    catch e
        if isa(e, InvalidStateException) && e.state==:closed
            return true
        else
            rethrow(e)
        end
    end
end
next(c::StackChannel{T}, state) where {T} = (v=get(state[]); state[]=nothing; (v, state))

iteratorsize(::Type{C}) where {C<:StackChannel} = SizeUnknown()
