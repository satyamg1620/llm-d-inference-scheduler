/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package internal

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// FinalState encapsulates the terminal outcome of a FlowItem's lifecycle.
type FinalState struct {
	Outcome types.QueueOutcome
	Err     error
}

// FlowItem is the internal representation of a request managed by the Flow Controller.
//
// # Lifecycle Management
//
// Finalization (determining outcome) can be initiated by the Controller (e.g., Context expiry) or the Processor (e.g.,
// Dispatch/Reject). It sets the outcome and signals the waiting goroutine.
//
// # Synchronization
//
// Atomic operations synchronize state across the Controller and Processor goroutines:
//   - finalState (atomic.Pointer): Safely publishes the outcome.
//   - handle (atomic.Pointer): Safely publishes the queue admission status.
type FlowItem struct {
	// --- Immutable fields during a single lifecycle ---

	enqueueTime     time.Time
	effectiveTTL    time.Duration
	noEndpointsTTL  time.Duration
	originalRequest flowcontrol.FlowControlRequest

	// --- Synchronized State ---

	// handle stores the types.QueueItemHandle atomically.
	// Written by the Processor (SetHandle) when admitted.
	// Read by inferOutcome (called by Finalize) to infer the outcome (Rejected vs. Evicted).
	// Distinguishing between pre-admission (Rejection) and post-admission (Eviction) during asynchronous finalization
	// relies on whether this handle is nil or non-nil.
	handle atomic.Pointer[flowcontrol.QueueItemHandle]

	// finalState holds the result of the finalization. Stored atomically once.
	// Use FinalState() for safe access.
	finalState atomic.Pointer[FinalState]

	// --- Finalization Signaling ---

	// done is the channel used to signal the completion of the item's lifecycle.
	// Buffered to size 1 to prevent Finalize from blocking.
	done chan *FinalState

	// onceFinalize ensures the finalization logic runs exactly once per lifecycle.
	onceFinalize sync.Once
}

var _ flowcontrol.QueueItemAccessor = &FlowItem{}

// NewItem allocates and initializes a new FlowItem for a request lifecycle.
//
// The item carries one queue-wait budget per unavailability regime: effectiveTTL applies while the candidate pool has
// endpoints, noEndpointsTTL while it is empty. Which one binds is resolved by ExpiredState on every expiry scan rather
// than fixed here, because a pool going from empty to non-empty is the entire point of scale-from-zero.
func NewItem(
	req flowcontrol.FlowControlRequest,
	effectiveTTL, noEndpointsTTL time.Duration,
	enqueueTime time.Time,
) *FlowItem {
	return &FlowItem{
		enqueueTime:     enqueueTime,
		effectiveTTL:    effectiveTTL,
		noEndpointsTTL:  noEndpointsTTL,
		originalRequest: req,
		done:            make(chan *FinalState, 1),
	}
}

// EnqueueTime returns the time the item was logically accepted by the FlowController.
func (fi *FlowItem) EnqueueTime() time.Time { return fi.enqueueTime }

// EffectiveTTL returns the time-to-live assigned to this item for the saturation regime, which is the budget that
// binds whenever the candidate pool has endpoints. Ordering policies use it to derive an absolute deadline.
func (fi *FlowItem) EffectiveTTL() time.Duration { return fi.effectiveTTL }

// NoEndpointsTTL returns the time-to-live assigned to this item for the empty-pool regime.
func (fi *FlowItem) NoEndpointsTTL() time.Duration { return fi.noEndpointsTTL }

// ExpiredState returns the terminal state for an item whose queue-wait budget is exhausted as of now, or nil if the
// item may keep waiting. A zero budget for the regime in effect disables expiry for that regime.
//
// While the pool is empty, the no-endpoint budget runs from the enqueue time: waiting is the only path to success, so
// the whole wait is charged to the cold start. While the pool has endpoints, the saturation budget runs from the later
// of the enqueue time and poolNonEmptySince, the moment the pool last became non-empty. Charging from the transition
// grants a request that waited out a cold start a full saturation budget in which to dispatch, instead of shedding it
// the instant it becomes servable against a budget it had already exhausted while nothing could serve it.
//
// A pool that flaps empty restarts the saturation budget on each transition, so the caller's context deadline remains
// the outer bound on total queue wait.
func (fi *FlowItem) ExpiredState(now time.Time, poolEmpty bool, poolNonEmptySince time.Time) *FinalState {
	if poolEmpty {
		if fi.noEndpointsTTL <= 0 || now.Sub(fi.enqueueTime) < fi.noEndpointsTTL {
			return nil
		}
		return &FinalState{
			Outcome: types.QueueOutcomeEvictedNoEndpointsTTL,
			Err:     fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrNoEndpoints),
		}
	}

	if fi.effectiveTTL <= 0 {
		return nil
	}
	chargeFrom := fi.enqueueTime
	if poolNonEmptySince.After(chargeFrom) {
		chargeFrom = poolNonEmptySince
	}
	if now.Sub(chargeFrom) < fi.effectiveTTL {
		return nil
	}
	return &FinalState{
		Outcome: types.QueueOutcomeEvictedTTL,
		Err:     fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrTTLExpired),
	}
}

// OriginalRequest returns the original FlowControlRequest object.
func (fi *FlowItem) OriginalRequest() flowcontrol.FlowControlRequest { return fi.originalRequest }

// Done returns a read-only channel that will receive the FinalState pointer exactly once.
func (fi *FlowItem) Done() <-chan *FinalState { return fi.done }

// FinalState returns the FinalState if the item has been finalized, or nil otherwise.
// Safe for concurrent access.
func (fi *FlowItem) FinalState() *FinalState { return fi.finalState.Load() }

// Handle returns the QueueItemHandle for this item within a queue.
// Returns nil if the item is not in a queue. Safe for concurrent access.
func (fi *FlowItem) Handle() flowcontrol.QueueItemHandle {
	ptr := fi.handle.Load()
	if ptr == nil {
		return nil
	}
	return *ptr
}

// SetHandle associates a QueueItemHandle with this item. Called by the queue implementation (via Processor).
// Safe for concurrent access.
func (fi *FlowItem) SetHandle(handle flowcontrol.QueueItemHandle) { fi.handle.Store(&handle) }

// Finalize determines the item's terminal state based on the provided cause (e.g., Context error) and the item's
// current admission status (queued or not).
//
// This method is intended for asynchronous finalization initiated by the Controller (e.g., TTL expiry).
// It is idempotent.
func (fi *FlowItem) Finalize(cause error) {
	fi.onceFinalize.Do(func() {
		// Atomically load the handle to determine if the item was admitted to a queue.
		// This synchronization is critical for correctly inferring the outcome across goroutines.
		isQueued := fi.Handle() != nil
		outcome, finalErr := inferOutcome(cause, isQueued)
		fi.finalizeInternal(outcome, finalErr)
	})
}

// FinalizeWithOutcome sets the item's terminal state explicitly.
//
// This method is intended for synchronous finalization by the Processor (Dispatch, Reject) or the Controller
// (Distribution failure).
// It is idempotent.
func (fi *FlowItem) FinalizeWithOutcome(outcome types.QueueOutcome, err error) {
	fi.onceFinalize.Do(func() {
		fi.finalizeInternal(outcome, err)
	})
}

// finalizeInternal is the core finalization logic. It must be called within the sync.Once.Do block.
// It captures the state, stores it atomically, and signals the Done channel.
func (fi *FlowItem) finalizeInternal(outcome types.QueueOutcome, err error) {
	finalState := &FinalState{
		Outcome: outcome,
		Err:     err,
	}

	// Atomically store the pointer. This is the critical memory barrier that publishes the state safely.
	fi.finalState.Store(finalState)

	duration := time.Since(fi.enqueueTime)
	flowKey := fi.originalRequest.FlowKey()
	metrics.RecordFlowControlRequestQueueDuration(
		flowKey.ID, strconv.Itoa(flowKey.Priority), outcome.String(),
		fi.originalRequest.InferencePoolName(),
		fi.OriginalRequest().ModelName(), fi.OriginalRequest().TargetModelName(),
		duration)

	fi.done <- finalState
	close(fi.done)
}

// inferOutcome determines the correct QueueOutcome and Error based on the cause of finalization and whether the item
// was already admitted to a queue.
func inferOutcome(cause error, isQueued bool) (types.QueueOutcome, error) {
	var specificErr error
	var outcomeIfEvicted types.QueueOutcome
	switch {
	case errors.Is(cause, types.ErrTTLExpired) || errors.Is(cause, context.DeadlineExceeded):
		specificErr = types.ErrTTLExpired
		outcomeIfEvicted = types.QueueOutcomeEvictedTTL
	case errors.Is(cause, context.Canceled):
		specificErr = fmt.Errorf("%w: %w", types.ErrContextCancelled, cause)
		outcomeIfEvicted = types.QueueOutcomeEvictedContextCancelled
	default:
		// Handle other potential causes (e.g., custom context errors).
		specificErr = cause
		outcomeIfEvicted = types.QueueOutcomeEvictedOther
	}

	if isQueued {
		// The item was in the queue when it expired/cancelled.
		return outcomeIfEvicted, fmt.Errorf("%w: %w", types.ErrEvicted, specificErr)
	}

	// The item was not yet in the queue (e.g., buffered in enqueueChan).
	// We treat this as a rejection, as it never formally consumed queue capacity.
	return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, specificErr)
}
