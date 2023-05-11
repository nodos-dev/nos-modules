//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// We deliberately use assert in here:
//
// boost-no-inspect

#ifndef BOOST_MATH_TOOLS_ASSERT_HPP
#define BOOST_MATH_TOOLS_ASSERT_HPP

#include <boost/math/tools/is_standalone.hpp>

#include <cassert>
#define BOOST_MATH_ASSERT(expr) assert(expr)
#define BOOST_MATH_ASSERT_MSG(expr, msg) assert((expr)&&(msg))
#define BOOST_MATH_STATIC_ASSERT(expr) static_assert(expr, #expr " failed")
#define BOOST_MATH_STATIC_ASSERT_MSG(expr, msg) static_assert(expr, msg)


#endif // BOOST_MATH_TOOLS_ASSERT_HPP
