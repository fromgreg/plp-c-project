#include "autograd.h"

#include <stdint.h>
#include <stdlib.h>

#include "util.h"

/* Growable array of tensor pointers
 *
 * The number of nodes in a graph is not known before it has been walked, so
 * the array grows on demand. Doubling the capacity keeps the total cost of n
 * pushes proportional to n. */

typedef struct {
    Tensor **items; /* array of Tensor pointers, hence pointer to pointer */
    size_t count;
    size_t capacity;
} NodeList;

static void node_list_push(NodeList *list, Tensor *t)
{
    if (list->count == list->capacity) {
        list->capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        /* With plain realloc this line would be a bug: on failure it returns
         * NULL, and overwriting items with NULL would leak the old block.
         * xrealloc aborts instead of returning NULL, so assigning directly is
         * safe. realloc(NULL, n) behaves like malloc(n), which covers the
         * first push. */
        list->items = xrealloc(list->items, list->capacity * sizeof *list->items);
    }
    list->items[list->count++] = t;
}

/* Topological sort
 *
 * Depth-first search from the root. A node is appended only after all of its
 * parents (post-order), so parents always come before their children.
 *
 * A node can be reachable along several paths (a diamond: one tensor feeds two
 * ops whose results meet again). visit_mark makes sure it is still appended
 * only once. Instead of a "visited" flag that would have to be cleared after
 * every sort, each sort gets its own number (epoch), and a node counts as
 * visited if its mark equals the current epoch. Marks left over from earlier
 * sorts are simply different numbers. */

/* static at file scope: private to this file, and it keeps its value between
 * calls. Starts at 0, which is also the mark of a fresh tensor. */
static uint32_t current_epoch;

static void visit(Tensor *t, uint32_t epoch, NodeList *order)
{
    if (t->visit_mark == epoch)
        return;
    t->visit_mark = epoch;
    for (int i = 0; i < t->n_parents; i++)
        visit(t->parents[i], epoch, order);
    node_list_push(order, t);
}

Tensor **graph_topo_order(Tensor *root, size_t *count)
{
    /* Unsigned overflow wraps around to 0. Skip 0, because a fresh tensor has
     * mark 0 and would look as if it had been visited already. */
    current_epoch++;
    if (current_epoch == 0)
        current_epoch = 1;

    NodeList order = {0}; /* items NULL, count 0, capacity 0 */
    visit(root, current_epoch, &order);
    *count = order.count;
    return order.items;
}

/* Backward pass and teardown */

void graph_backward(Tensor *root)
{
    if (root->grad == NULL)
        fatal("graph_backward: root has no grad buffer (a plain leaf)");

    size_t count;
    Tensor **order = graph_topo_order(root, &count);

    size_t n = tensor_size(root);
    for (size_t i = 0; i < n; i++)
        root->grad[i] = 1.0;

    /* Reverse topological order: a node is processed only after all the ops
     * that consume it, so its grad is complete before it is passed on to its
     * parents. i is unsigned, so the loop tests before it decrements. */
    for (size_t i = count; i-- > 0;) {
        Tensor *t = order[i];
        if (t->backward != NULL)
            t->backward(t);
    }

    free(order);
}

void graph_free(Tensor *root)
{
    /* Collect all nodes first, then free. Freeing during the walk would follow
     * parents pointers out of freed nodes, and a node that is reachable along
     * two paths would be freed twice. The array holds each node exactly once. */
    size_t count;
    Tensor **order = graph_topo_order(root, &count);

    for (size_t i = 0; i < count; i++)
        if (order[i]->kind == TENSOR_INTERMEDIATE)
            tensor_free(order[i]);

    free(order);
}
