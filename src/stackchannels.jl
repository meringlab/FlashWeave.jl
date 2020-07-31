# This file is altered from "base/channels.jl" and "stdlib/Distributed/src/remotecall.jl" distributed with Julia (License is MIT: https://julialang.org/license)
import Base: AbstractChannel, put!, take!, isready, isopen, close, lock, unlock, check_channel_state, notify, isbuffered
import Distributed: RemoteValue, RemoteException, SyncTake, call_on_owner, lookup_ref

stacktake!(c::Channel) = isbuffered(c) ? stacktake_buffered(c) : stacktake_unbuffered(c)
function stacktake_buffered(c::Channel)
    lock(c)
    try
        while isempty(c.data)
            check_channel_state(c)
            wait(c.cond_take)
        end
        v = pop!(c.data)
        notify(c.cond_put, nothing, false, false) # notify only one, since only one slot has become available for a put!.
        return v
    finally
        unlock(c)
    end
end

# 0-size channel
function stacktake_unbuffered(c::Channel{T}) where T
    lock(c)
    try
        check_channel_state(c)
        notify(c.cond_put, nothing, false, false)
        return wait(c.cond_take)::T
    finally
        unlock(c)
    end
end


stacktake!(rv::RemoteValue, args...) = stacktake!(rv.c, args...)
function stacktake_ref(rid, caller, args...)
    rv = lookup_ref(rid)
    synctake = false
    if myid() != caller && rv.synctake !== nothing
        # special handling for local put! / remote take! on unbuffered channel
        # github issue #29932
        synctake = true
        lock(rv.synctake)
    end

    v=take!(rv, args...)
    isa(v, RemoteException) && (myid() == caller) && throw(v)

    if synctake
        return SyncTake(v, rv)
    else
        return v
    end
end

"""
    take!(rr::RemoteChannel, args...)
Fetch value(s) from a [`RemoteChannel`](@ref) `rr`,
removing the value(s) in the process.
"""
stacktake!(rr::RemoteChannel, args...) = call_on_owner(stacktake_ref, rr, myid(), args...)
