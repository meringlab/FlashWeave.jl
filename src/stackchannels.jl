# This file is altered from "base/channels.jl" distributed with Julia (License is MIT: https://julialang.org/license

import Base: AbstractChannel, put!, push!, pop!, take!, wait, fetch, eltype, show, close

"""
    StackChannel{T}(sz::Int)
Constructs a `Channel` with an internal buffer that can hold a maximum of `sz` objects
of type `T`.
[`put!`](@ref) calls on a full StackChannel block until an object is removed with [`take!`](@ref).
`Channel(0)` constructs an unbuffered StackChannel. `put!` blocks until a matching `take!` is called.
And vice-versa.
Other constructors:
* `Channel(Inf)`: equivalent to `Channel{Any}(typemax(Int))`
* `Channel(sz)`: equivalent to `Channel{Any}(sz)`
"""
mutable struct StackChannel{T} <: AbstractChannel{T}
    cond_take::Condition                 # waiting for data to become available
    cond_put::Condition                  # waiting for a writeable slot
    state::Symbol
    excp::Union{Exception, Nothing}         # exception to be thrown when state != :open

    data::Vector{T}
    sz_max::Int                          # maximum size of StackChannel

    # Used when sz_max == 0, i.e., an unbuffered StackChannel.
    waiters::Int
    takers::Vector{Task}
    putters::Vector{Task}

    function StackChannel{T}(sz::Float64) where T
        if sz == Inf
            StackChannel{T}(typemax(Int))
        else
            StackChannel{T}(convert(Int, sz))
        end
    end
    function StackChannel{T}(sz::Integer) where T
        if sz < 0
            throw(ArgumentError("Channel size must be either 0, a positive integer or Inf"))
        end
        ch = new(Condition(), Condition(), :open, nothing, Vector{T}(), sz, 0)
        if sz == 0
            ch.takers = Vector{Task}()
            ch.putters = Vector{Task}()
        end
        return ch
    end
end

StackChannel(sz) = StackChannel{Any}(sz)

# special constructors
"""
    StackChannel(func::Function; ctype=Any, csize=0, taskref=nothing)
Create a new task from `func`, bind it to a new StackChannel of type
`ctype` and size `csize`, and schedule the task, all in a single call.
`func` must accept the bound StackChannel as its only argument.
If you need a reference to the created task, pass a `Ref{Task}` object via
keyword argument `taskref`.
Return a `Channel`.
# Examples
```jldoctest
julia> chnl = StackChannel(c->foreach(i->put!(c,i), 1:4));
julia> typeof(chnl)
Channel{Any}
julia> for i in chnl
           @show i
       end;
i = 1
i = 2
i = 3
i = 4
```
Referencing the created task:
```jldoctest
julia> taskref = Ref{Task}();
julia> chnl = StackChannel(c->(@show take!(c)); taskref=taskref);
julia> istaskdone(taskref[])
false
julia> put!(chnl, "Hello");
take!(c) = "Hello"
julia> istaskdone(taskref[])
true
```
"""
function StackChannel(func::Function; ctype=Any, csize=0, taskref=nothing)
    chnl = StackChannel{ctype}(csize)
    task = Task(() -> func(chnl))
    bind(chnl, task)
    yield(task) # immediately start it

    isa(taskref, Ref{Task}) && (taskref[] = task)
    return chnl
end


closed_exception() = InvalidStateException("Channel is closed.", :closed)

isbuffered(c::StackChannel) = c.sz_max==0 ? false : true

function check_channel_state(c::StackChannel)
    if !isopen(c)
        c.excp !== nothing && throw(c.excp)
        throw(closed_exception())
    end
end
"""
    close(c::StackChannel)
Close a StackChannel. An exception is thrown by:
* [`put!`](@ref) on a closed StackChannel.
* [`take!`](@ref) and [`fetch`](@ref) on an empty, closed StackChannel.
"""
function close(c::StackChannel)
    c.state = :closed
    c.excp = closed_exception()
    notify_error(c)
    nothing
end
isopen(c::StackChannel) = (c.state == :open)

"""
    bind(chnl::StackChannel, task::Task)
Associate the lifetime of `chnl` with a task.
`Channel` `chnl` is automatically closed when the task terminates.
Any uncaught exception in the task is propagated to all waiters on `chnl`.
The `chnl` object can be explicitly closed independent of task termination.
Terminating tasks have no effect on already closed `Channel` objects.
When a StackChannel is bound to multiple tasks, the first task to terminate will
close the StackChannel. When multiple StackChannels are bound to the same task,
termination of the task will close all of the bound StackChannels.
# Examples
```jldoctest
julia> c = StackChannel(0);
julia> task = @async foreach(i->put!(c, i), 1:4);
julia> bind(c,task);
julia> for i in c
           @show i
       end;
i = 1
i = 2
i = 3
i = 4
julia> isopen(c)
false
```
```jldoctest
julia> c = StackChannel(0);
julia> task = @async (put!(c,1);error("foo"));
julia> bind(c,task);
julia> take!(c)
1
julia> put!(c,1);
ERROR: foo
Stacktrace:
[...]
```
"""
function bind(c::StackChannel, task::Task)
    ref = WeakRef(c)
    register_taskdone_hook(task, tsk->close_chnl_on_taskdone(tsk, ref))
    c
end

"""
    channeled_tasks(n::Int, funcs...; ctypes=fill(Any,n), csizes=fill(0,n))
A convenience method to create `n` StackChannels and bind them to tasks started
from the provided functions in a single call. Each `func` must accept `n` arguments
which are the created StackChannels. StackChannel types and sizes may be specified via
keyword arguments `ctypes` and `csizes` respectively. If unspecified, all StackChannels are
of type `Channel{Any}(0)`.
Returns a tuple, `(Array{StackChannel}, Array{Task})`, of the created StackChannels and tasks.
"""
function channeled_tasks(n::Int, funcs...; ctypes=fill(Any,n), csizes=fill(0,n))
    @assert length(csizes) == n
    @assert length(ctypes) == n

    chnls = map(i -> StackChannel{ctypes[i]}(csizes[i]), 1:n)
    tasks = Task[ Task(() -> f(chnls...)) for f in funcs ]

    # bind all tasks to all StackChannels and schedule them
    foreach(t -> foreach(c -> bind(c, t), chnls), tasks)
    foreach(schedule, tasks)
    yield() # Allow scheduled tasks to run

    return (chnls, tasks)
end

function close_chnl_on_taskdone(t::Task, ref::WeakRef)
    if ref.value !== nothing
        c = ref.value
        !isopen(c) && return
        if istaskfailed(t)
            c.state = :closed
            c.excp = task_result(t)
            notify_error(c)
        else
            close(c)
        end
    end
end

struct InvalidStateException <: Exception
    msg::AbstractString
    state::Symbol
end

"""
    put!(c::StackChannel, v)
Append an item `v` to the StackChannel `c`. Blocks if the StackChannel is full.
For unbuffered StackChannels, blocks until a [`take!`](@ref) is performed by a different
task.
"""
function put!(c::StackChannel, v)
    check_channel_state(c)
    isbuffered(c) ? put_buffered(c,v) : put_unbuffered(c,v)
end

function put_buffered(c::StackChannel, v)
    while length(c.data) == c.sz_max
        wait(c.cond_put)
    end
    push!(c.data, v)

    # notify all, since some of the waiters may be on a "fetch" call.
    notify(c.cond_take, nothing, true, false)
    v
end

function put_unbuffered(c::StackChannel, v)
    if length(c.takers) == 0
        push!(c.putters, current_task())
        c.waiters > 0 && notify(c.cond_take, nothing, false, false)

        try
            wait()
        catch ex
            filter!(x->x!=current_task(), c.putters)
            rethrow(ex)
        end
    end
    taker = pop!(c.takers)
    yield(taker, v) # immediately give taker a chance to run, but don't block the current task
    return v
end

push!(c::StackChannel, v) = put!(c, v)

"""
    fetch(c::StackChannel)
Wait for and get the first available item from the StackChannel. Does not
remove the item. `fetch` is unsupported on an unbuffered (0-size) StackChannel.
"""
fetch(c::StackChannel) = isbuffered(c) ? fetch_buffered(c) : fetch_unbuffered(c)
function fetch_buffered(c::StackChannel)
    wait(c)
    c.data[1]
end
fetch_unbuffered(c::StackChannel) = throw(ErrorException("`fetch` is not supported on an unbuffered StackChannel."))


"""
    take!(c::StackChannel)
Remove and return a value from a [`Channel`](@ref). Blocks until data is available.
For unbuffered StackChannels, blocks until a [`put!`](@ref) is performed by a different
task.
"""
take!(c::StackChannel) = isbuffered(c) ? take_buffered(c) : take_unbuffered(c)
function take_buffered(c::StackChannel)
    wait(c)
    v = pop!(c.data)
    notify(c.cond_put, nothing, false, false) # notify only one, since only one slot has become available for a put!.
    v
end

pop!(c::StackChannel) = take!(c)

# 0-size StackChannel
function take_unbuffered(c::StackChannel{T}) where T
    check_channel_state(c)
    push!(c.takers, current_task())
    try
        if length(c.putters) > 0
            let refputter = Ref(pop!(c.putters))
                return Base.try_yieldto(refputter) do putter
                    # if we fail to start putter, put it back in the queue
                    putter === current_task || push!(c.putters, putter)
                end::T
            end
        else
            return wait()::T
        end
    catch ex
        filter!(x->x!=current_task(), c.takers)
        rethrow(ex)
    end
end

"""
    isready(c::StackChannel)
Determine whether a [`Channel`](@ref) has a value stored to it. Returns
immediately, does not block.
For unbuffered StackChannels returns `true` if there are tasks waiting
on a [`put!`](@ref).
"""
isready(c::StackChannel) = n_avail(c) > 0
n_avail(c::StackChannel) = isbuffered(c) ? length(c.data) : length(c.putters)

wait(c::StackChannel) = isbuffered(c) ? wait_impl(c) : wait_unbuffered(c)
function wait_impl(c::StackChannel)
    while !isready(c)
        check_channel_state(c)
        wait(c.cond_take)
    end
    nothing
end

function wait_unbuffered(c::StackChannel)
    c.waiters += 1
    try
        wait_impl(c)
    finally
        c.waiters -= 1
    end
    nothing
end

function notify_error(c::StackChannel, err)
    notify_error(c.cond_take, err)
    notify_error(c.cond_put, err)

    # release tasks on a `wait()/yieldto()` call (on unbuffered StackChannels)
    if !isbuffered(c)
        waiters = filter!(t->(t.state == :runnable), vcat(c.takers, c.putters))
        foreach(t->schedule(t, err; error=true), waiters)
    end
end
notify_error(c::StackChannel) = notify_error(c, c.excp)

eltype(::Type{StackChannel{T}}) where {T} = T

show(io::IO, c::StackChannel) = print(io, "$(typeof(c))(sz_max:$(c.sz_max),sz_curr:$(n_avail(c)))")

function iterate(c::StackChannel, state=nothing)
    try
        return (take!(c), nothing)
    catch e
        if isa(e, InvalidStateException) && e.state==:closed
            return nothing
        else
            rethrow(e)
        end
    end
end

IteratorSize(::Type{<:StackChannel}) = SizeUnknown()
