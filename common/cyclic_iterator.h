#ifndef CYCLIC_ITERATOR_H
#define CYCLIC_ITERATOR_H

#include <thrust/iterator/iterator_adaptor.h>
// derive cyclic_iterator from iterator_adaptor
template<typename Iterator>
  class cyclic_iterator
    : public thrust::iterator_adaptor<
        cyclic_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
        Iterator                   // the second template parameter is the name of the iterator we're adapting
                                   // we can use the default for the additional template parameters
      >
{
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<
      cyclic_iterator<Iterator>,
      Iterator
    > super_t;
    __host__ __device__
    cyclic_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
  private:
    // repeat each element of the adapted range n times
    unsigned int n;
    // used to keep track of where we began
    const Iterator begin;
    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
      return *(begin + (this->base() - begin) % n);
    }
};

#endif /* CYCLIC_ITERATOR_H */
