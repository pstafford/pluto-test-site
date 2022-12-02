### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 81bf285c-475d-466a-846f-82db6f6de9ab
begin
	using PlutoUI
	using Plots
	using StatsPlots
	using ColorSchemes
	using Distributions
	using HypertextLiteral
	using QuadGK
	
	TableOfContents(depth=4)
end

# ╔═╡ afa78dcf-24b9-489e-a10b-907e2fb2452e
include("test.jl");

# ╔═╡ b32f2a7e-3dad-11ec-3854-773f1ca4cbbc
md"""
# PSHA Implementation Examples
Hazard analysis for a simplified area source and strike-slip fault source.

The annual rate of exceedance $\lambda(IM>im)$ is defined as:
```math
\lambda(IM>im) = \sum_{i=1}^{n_{rup}} P\left( IM>im \mid rup, site \right)\lambda(rup_i)
```

As such, we need to specify the rate of occurrence for each rupture represented by the source model, $\lambda(rup_i)$, and also need a ground-motion model that provides the conditional probability of exceedance $P(IM>im\mid rup, site)$.
"""

# ╔═╡ 87178931-1277-4c5a-b718-b15335a0622f
md"""
## Magnitude-Frequency Distributions
Examples are provided using both the Gutenberg-Richter distribution and the Characteristic distribution (as defined by Youngs & Coppersmith, 1985).

Within the hazard calculations, it is convenient to be able to define methods that are agonostic to the type of distribution that we are working with.
In our code we therefore define an abstract type called `MagnitudeFrequencyDistribution` and then define concrete types of `GutenbergRichter` and `YoungsCoppersmith` that implement the details of the respective distributions. 
"""

# ╔═╡ 5f514ffb-ad02-48b3-a2f4-1ea974562b03
"""
	abstract type MagnitudeFrequencyDistribution end

Abstract type representing a generic magnitude-frequency distribution.
Defined only to permit subsequent functions to be defined for any concrete magnitude-frequency distribution type
"""
abstract type MagnitudeFrequencyDistribution end

# ╔═╡ a6e100e1-39e8-4f2e-81b0-5b5d0982330c
md"""
### Gutenberg Richter Distribution
First we define a custom type `GutenbergRichter` that simply captures the parameters of the Gutenberg-Richter distribution (the doubly-bounded exponential distribution)
"""

# ╔═╡ 436c894f-7864-433f-86b8-6d3827290aa9
"""
	GutenbergRichter{T} <: MagnitudeFrequencyDistribution where T<:Real

Custom type representing the parameters of the Gutenberg-Richter distribution, holding the following parameters:
- `m_min`: the minimum magnitude
- `m_max`: the maximum magnitude
- `b_value`: the _b_-value
- `λ_eq`: the total annual rate of earthquakes of magnitude ≥ `m_min`
"""
struct GutenbergRichter{T} <: MagnitudeFrequencyDistribution where T<:Real
    m_min::T
    m_max::T
    b_value::T
    λ_eq::T
end

# ╔═╡ 891dc275-7869-4c20-99db-f7240363af4f
md"""
#### Probability density function
The probability density function (pdf) for the Gutenberg-Richter (GR) distribtuion is defined as:
```math
f_M(m) = \begin{cases}
	0 & \text{for } m < m_{\min} \text{ or } m > m_{\max} \\
	\frac{\beta \exp\left[ -\beta\left(m-m_{\min}\right)\right]}{1-\exp\left[ -\beta\left(m_{\max}-m_{\min}\right) \right]} & \text{otherwise} \\
\end{cases}
```

This method is implemented as follows:
"""

# ╔═╡ 7ea16336-3500-4214-8ba6-6b459da07f6c
"""
	pdf(gr::GutenbergRichter, m::T) where T<:Real

Probability density function (pdf) for the Gutenberg-Richter distribution.

- `gr` is an instance of a `GutenbergRichter` distribution
- `m` is the magnitude for which the pdf should be evaluated
"""
function pdf(gr::GutenbergRichter, m::T) where T<:Real
    if m < gr.m_min || m > gr.m_max
        return zero(typeof(m))
    else
        β = gr.b_value * log(10.0)
        return β * exp( -β*( m - gr.m_min ) ) / ( 1.0 - exp(-β*(gr.m_max - gr.m_min)) )
    end
end

# ╔═╡ f8215680-b83c-4545-8e1d-c21f898a0eae
md"""
#### Cumulative distribution function
The cumulative distribution function (cdf) for the GR distribution is defined as:
```math
F_M(m) = \begin{cases}
0 & \text{for } m \le m_{\min} \\
\frac{1-\exp\left[-\beta\left(m-m_{\min}\right)\right]}{1-\exp\left[ -\beta\left(m_{\max}-m_{\min}\right)\right]} & \text{for } m_{\min} < m < m_{\max} \\
1 & \text{for } m \ge m_{\max} \\
\end{cases}
```
The method is implemented as follows:
"""

# ╔═╡ e0dc1e81-037d-4828-830d-9085998cc356
"""
	cdf(gr::GutenbergRichter, m::T) where T<:Real

Cumulative distribution function (cdf) for the Gutenberg-Richter distribution.

- `gr` is an instance of a `GutenbergRichter` distribution
- `m` is the magnitude for which the cdf should be evaluated
"""
function cdf(gr::GutenbergRichter, m::T) where T <: Real
    if m < gr.m_min
        return zero(typeof(m))
    elseif m > gr.m_max
        return one(typeof(m))
    else
        β = gr.b_value * log(10.0)
        return (1.0 - exp(-β*(m - gr.m_min)))/(1.0 - exp(-β*(gr.m_max - gr.m_min)))
    end
end


# ╔═╡ 97e6a615-88c5-48aa-a8a7-b22221c8be91
md"""
#### Complementary cumulative distribution function
In addition to the cdf, we define its compliment (ccdf) to readily enable probability of exceedance calculations to be performed (if desired).
The ccdf is defined in terms of the cdf via:
```math
G_M(m) = 1 - F_M(m)
```
and is implemented as follows:
"""

# ╔═╡ c890bc4c-3a20-4511-9e74-aa48a3a78e01
"""
	ccdf(gr::GutenbergRichter, m::T) where T<:Real

Complementary cumulative distribution function (ccdf) for the Gutenberg-Richter distribution.

- `gr` is an instance of a `GutenbergRichter` distribution
- `m` is the magnitude for which the ccdf should be evaluated
"""
function ccdf(gr::GutenbergRichter, m::T) where T <: Real
    return 1.0 - cdf(gr, m)
end


# ╔═╡ 58e3783d-c98b-4260-8e36-79d1c80fb7b2
md"""
### Characteristic Distribution (Youngs & Coppersmith, 1985)
Similarly to the GR distribution, we implement equivalent methods related to the charcteristic distribution in this section, starting with the definition of the following concrete type.
"""

# ╔═╡ ba3c7dba-34dd-45a9-b8bf-bcc51d7fb625
"""
	YoungsCoppersmith{T} <: MagnitudeFrequencyDistribution where T <: Real

Custom type representing the parameters of the Youngs & Coppersmith characteristic earthquake magnitude-frequecny distribution. The type holds the following parameters:
- `m_min`: the minimum magnitude
- `m_max`: the maximum magnitude
- `Δm_char`: the width of the uniform distribution representing characteristic events
- `p_char`: the probability of a characteristic event
- `b_value`: the _b_-value for the exponential portion of the distribution
- `λ_eq`: the total annual rate of earthquakes of magnitude ≥ `m_min`

"""
struct YoungsCoppersmith{T} <: MagnitudeFrequencyDistribution where T <: Real
    m_min::T
    m_char::T
    Δm_char::T
    p_char::T
    b_value::T
    λ_eq::T
end

# ╔═╡ b570c8d4-c2f8-4dcb-b623-b1c1ea28026a
md"""
#### Probability density function
The probability density function for the characteristic earthquake distribution is defined as:
```math
f_M(m) = \begin{cases}
	0 & \text{for } m < m_{\min} \text{ or } m > m_{char}^{HI} \\
	\left( 1-P_C \right) \frac{\beta \exp\left[ -\beta\left(m-m_{\min}\right)\right]}{1-\exp\left[-\beta\left(m_{char}^{LO}-m_{\min}\right)\right]} & \text{for } m_{\min} \le m < m_{char}^{LO} \\
	P_C/\Delta m_{char} & \text{for } m_{char}^{LO} \le m \le m_{char}^{HI} \\
\end{cases}
```
where $P_{C}$ is the probability of characteristic earthquakes of any size, $m_{char}^{HI}$ is the maximum size of characteristic events, $m_{char}^{LO}$ is the minimum size of characteristic events (and the upper magnitude for the exponential branch of the distribution).

Note that a pdf function is also implemented that takes `YoungsCoppersmith` instances, and that the format is the same as for the previously defined pdf for the `GutenbergRichter` instances. 
That is, pdf values can be computed using `pdf(gr, m)` or `pdf(yc, m)`, where `gr` is an instance of a `GutenbergRichter` type, `yc` is an instance of a `YoungsCoppersmith` type, and `m` is an earthquake magnitude.
"""

# ╔═╡ bd36b64d-88ee-47ed-9509-5b05441abb2f
"""
	pdf(yc::YoungsCoppersmith, m::T) where T <: Real

Probability density function (pdf) for the Youngs & Coppersmith (1985) characteristic distribution.
- `yc` is an instance of `YoungsCoppermith` 
- `m` is the magnitude for which the pdf is to be evaluated
"""
function pdf(yc::YoungsCoppersmith, m::T) where T <: Real
    m_max = yc.m_char + yc.Δm_char/2
    if m < yc.m_min || m > m_max
        return zero(typeof(m))
    elseif m < yc.m_char - yc.Δm_char/2
        β = yc.b_value * log(10.0)
        return (1.0 - yc.p_char) * β * exp( -β*( m - yc.m_min ) ) / ( 1.0 - exp(-β*(yc.m_char - yc.Δm_char/2 - yc.m_min)) )
    else
        return yc.p_char / yc.Δm_char
    end
end;


# ╔═╡ 346722a4-a1cb-4697-b093-defd57b3c610
"""
	cdf(yc::YoungsCoppersmith, m::T) where T <: Real

Cumulative distribution function (cdf) for the Youngs & Coppersmith characteristic  distribution.

- `yc` is an instance of a `YoungsCoppersmith` distribution
- `m` is the magnitude for which the cdf should be evaluated
"""
function cdf(yc::YoungsCoppersmith, m::T) where T <: Real
    if m < yc.m_min
        return zero(typeof(m))
    elseif m > yc.m_char + yc.Δm_char/2
        return one(typeof(m))
    elseif m < yc.m_char - yc.Δm_char/2
        β = yc.b_value * log(10.0)
        return (1.0-yc.p_char)*((1.0 - exp(-β*(m - yc.m_min)))/(1.0 - exp(-β*(yc.m_char - yc.Δm_char/2 - yc.m_min))))
    else
        return (1.0-yc.p_char) + yc.p_char/yc.Δm_char * (m - (yc.m_char - yc.Δm_char/2))
    end
end;


# ╔═╡ c4e39f09-1ebf-4eff-8a3e-cc9f81c2b288
"""
	ccdf(yc::YoungsCoppersmith, m::T) where T <: Real

Complementary cumulative distribution function (cdf) for the Youngs & Coppersmith characteristic  distribution.

- `yc` is an instance of a `YoungsCoppersmith` distribution
- `m` is the magnitude for which the cdf should be evaluated
"""
function ccdf(yc::YoungsCoppersmith, m::T) where T <: Real
    return 1.0 - cdf(yc, m)
end;


# ╔═╡ 7be95d4b-ade1-48f6-8000-6e6e9409987e
md"""
### Rates of Exceedance and Occurrence
The distributions defined thus far specify outputs in terms of probabilities and probability densities, but for the hazard calculations we need to define rates of occurrence (or exceedance).

The specification of these rates is straightforward given the distributions we've already defined.
For example, the rate of exceedance is defined as:
```math
\lambda(M\ge m) = \lambda(M\ge m_{\min}) \times G_M(m)
```

The rate of occurrence could be approximated through the use of the pdf for a distribution, but as these are continuous distributions, the probability of occurrence of a particular magnitude is actually zero. 
Instead, when we talk about probability of occurrence, we are talking about the probability that a rupture occurs with a magnitude in a small range around some value of interest. 
Mathematically, we express this as:
```math
\lambda(M=m) \equiv \lambda\left[M\in \left(m-\frac{\Delta m}{2}, m+\frac{\Delta m}{2} \right) \right]
```
or
```math
\lambda(M=m) = \lambda(M\ge m_{\min}) \times \left[ F_M\left(m+\frac{\Delta m}{2}\right) - F_M\left(m-\frac{\Delta m}{2}\right) \right]
```
Of course, this is the same as:
```math
\lambda(M=m) = \lambda(M\ge m_{\min}) \times \left[ G_M\left(m-\frac{\Delta m}{2}\right) - G_M\left(m+\frac{\Delta m}{2}\right) \right]
```
"""

# ╔═╡ 61db3c2a-d98b-4246-84c3-8731cb29e669
"""
	rate_of_exceedance(dist::MagnitudeFrequencyDistribution, m::T) where T <: Real

Rate of exceedance of magnitude `m`, using a magnitude frequency distribution `dist`
- `dist` is a concrete instance of the abstract type `MagnitudeFrequencyDistribution`, which in practice means `GutenbergRichter` or `YoungsCoppersmith`

See also: [`ccdf`](@ref)
"""
function rate_of_exceedance(dist::MagnitudeFrequencyDistribution, m::T) where T <: Real
    return dist.λ_eq * ccdf(dist, m)
end;


# ╔═╡ 313c5c6b-2eff-4df5-adae-a01c4eb8de9b
"""
	rate_of_nonexceedance(dist::MagnitudeFrequencyDistribution, m::T) where T <: Real

Rate of nonexceedance, similar to [`rate_of_exceedance`](@ref), but making use of the cdf rather than ccdf function.
- `dist` is a concrete instance of the abstract type `MagnitudeFrequencyDistribution`, which in practice means `GutenbergRichter` or `YoungsCoppersmith`

See also: [`cdf`](@ref)
"""
function rate_of_nonexceedance(dist::MagnitudeFrequencyDistribution, m::T) where T <: Real
    return dist.λ_eq * cdf(dist, m)
end;


# ╔═╡ dd3c5633-7295-454b-be67-ea38ded8850c
"""
	rate_of_occurrence(dist::MagnitudeFrequencyDistribution, m_lo::T, m_hi::T) where T <: Real

Rate of occurrence, defined by the probability of a magnitude being in the interval between `m_lo` and `m_hi`

- `dist` is a concrete instance of the abstract type `MagnitudeFrequencyDistribution`, which in practice means `GutenbergRichter` or `YoungsCoppersmith`
- `m_lo` is the smaller magnitude defining the interval
- `m_hi` is the larger magnitude defining the interval
"""
function rate_of_occurrence(dist::MagnitudeFrequencyDistribution, m_lo::T, m_hi::T) where T <: Real
    return dist.λ_eq * ( cdf(dist, m_hi) - cdf(dist, m_lo) )
end


# ╔═╡ 0f1e9101-32ec-435e-ba2d-40d7215fe58a
md"""
## GMM: Abrahamson & Silva (1997)  

The Abrahamson & Silva (1997) ground-motion model specifies spectral accelerations for a range of periods as a function of magnitude, rupture distance, site class and style-of-faulting. The site scaling includes the effects of nonlinear site response, with amplification predicted as a function of the peak ground acceleration on rock.

This model is implemented below, and the details of the function are provided in the documentation. 
The function takes values (floating point numbers) for the period, `period`, magnitude `m`, and rupture distance `r`, and integer indicator variables to define the site conditions, the style-of-faulting, and whether the site is on the hanging wall.
The function returns the mean and standard deviation of the logarithmic spectral acceleration for the period requested.
"""

# ╔═╡ 0fc11681-3776-4521-a67d-2aa7f5dce5fe
begin
# Define the Abrahamson & Silva (1997) coefficients as constant arrays
const Ti = [ 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.1, 0.12, 0.15, 0.17, 0.2, 0.24, 0.3, 0.36, 0.4, 0.46, 0.5, 0.6, 0.75, 0.85, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0 ]
const c4i = [ 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.58, 5.54, 5.5, 5.39, 5.27, 5.19, 5.1, 4.97, 4.8, 4.62, 4.52, 4.38, 4.3, 4.12, 3.9, 3.81, 3.7, 3.55, 3.5, 3.5, 3.5, 3.5 ]
const a1i = [ 1.64, 1.64, 1.64, 1.69, 1.78, 1.87, 1.94, 2.037, 2.1, 2.16, 2.272, 2.407, 2.43, 2.406, 2.293, 2.114, 1.955, 1.86, 1.717, 1.615, 1.428, 1.16, 1.02, 0.828, 0.26, -0.15, -0.69, -1.13, -1.46 ]
const a2i = [ 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512, 0.512 ]
const a3i = [ -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.145, -1.135, -1.115, -1.079, -1.035, -1.0052, -0.988, -0.9652, -0.9515, -0.9218, -0.8852, -0.8648, -0.8383, -0.7721, -0.725, -0.725, -0.725, -0.725 ]
const a4i = [ -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144, -0.144 ]
const a5i = [ 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.592, 0.581, 0.557, 0.528, 0.512, 0.49, 0.438, 0.4, 0.4, 0.4, 0.4 ]
const a6i = [ 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.232, 0.198, 0.17, 0.154, 0.132, 0.119, 0.091, 0.057, 0.038, 0.013, -0.049, -0.094, -0.156, -0.2, -0.2 ]
const a9i = [ 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.331, 0.309, 0.281, 0.21, 0.16, 0.089, 0.039, 0 ]
const a10i = [ -0.417, -0.417, -0.417, -0.47, -0.555, -0.62, -0.665, -0.628, -0.609, -0.598, -0.591, -0.577, -0.522, -0.445, -0.35, -0.219, -0.123, -0.065, 0.02, 0.085, 0.194, 0.32, 0.37, 0.423, 0.6, 0.61, 0.63, 0.64, 0.664 ]
const a11i = [ -0.23, -0.23, -0.23, -0.23, -0.251, -0.267, -0.28, -0.28, -0.28, -0.28, -0.28, -0.28, -0.265, -0.245, -0.223, -0.195, -0.173, -0.16, -0.136, -0.121, -0.089, -0.05, -0.028, 0, 0.04, 0.04, 0.04, 0.04, 0.04 ]
const a12i = [ 0, 0, 0, 0.0143, 0.0245, 0.028, 0.03, 0.03, 0.03, 0.028, 0.018, 0.005, -0.004, -0.0138, -0.0238, -0.036, -0.046, -0.0518, -0.0594, -0.0635, -0.074, -0.0862, -0.0927, -0.102, -0.12, -0.14, -0.1726, -0.1956, -0.215 ]
const a13i = [ 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17 ]
const c1i = [ 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4 ]
const c5i = [ 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03 ]
const ni = [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]
const b5i = [ 0.7, 0.7, 0.7, 0.7, 0.71, 0.71, 0.72, 0.73, 0.74, 0.74, 0.75, 0.75, 0.76, 0.77, 0.77, 0.78, 0.79, 0.79, 0.8, 0.8, 0.81, 0.81, 0.82, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89 ]
const b6i = [ 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.132, 0.13, 0.127, 0.123, 0.121, 0.118, 0.11, 0.105, 0.097, 0.092, 0.087 ]

num_periods = length(Ti)

"""
	as1997_coefficients_for_index(idx::Int)

Returns the coefficients for a specified index. Used within other AS1997 functions
"""
function as1997_coefficients_for_index(idx::Int)
    return c4i[idx], a1i[idx], a2i[idx], a3i[idx], a4i[idx], a5i[idx], a6i[idx], a9i[idx], a10i[idx], a11i[idx], a12i[idx], a13i[idx], c1i[idx], c5i[idx], ni[idx], b5i[idx], b6i[idx]
end


"""
	as1997_pga_rock(m::T, r::T, faultType::Int, hangingWall::Int=0) where T<:Real

Predicted peak ground acceleration on rock using the Abrahamson & Silva (1997) GMM. Note that this metric is used within their model for predicting the effects of nonlinear site response for soil sites.

Inputs:
- `m` is the magnitude
- `r` is the rupture distance [km]
- `faultType` is 2 for reverse, 1 for reverse/oblique, 0 otherwise
- `hangingWall` equal to 1 if site is on the hanging wall, 0 otherwise

Returns:
- `pga_rock` the peak ground acceleration on rock [g]
"""
function as1997_pga_rock(m::T, r::T, faultType::Int, hangingWall::Int=0) where T<:Real
    c4, a1, a2, a3, a4, a5, a6, a9, a10, a11, a12, a13, c1, c5, n, b5, b6 = as1997_coefficients_for_index(1)

    if m <= c1
        f1 = a1 + a2*(m - c1) + a12*(8.5 - m)^n + (a3 + a13*(m - c1))*log(sqrt(r^2 + c4^2))
    else
        f1 = a1 + a4*(m - c1) + a12*(8.5 - m)^n + (a3 + a13*(m - c1))*log(sqrt(r^2 + c4^2))
    end

    if faultType != 0
        if m <= 5.8
            f3 = a5
        elseif m < c1
            f3 = a5 + (a6 - a5)*(m - 5.8)/(c1 - 5.8)
        else
            f3 = a6
        end
        f3 *= faultType/2.0
    else
        f3 = 0.0
    end

    if hangingWall != 0
        if m <= 5.5
            fhwM = 0.0
        elseif m < 6.5
            fhwM = m - 5.5
        else
            fhwM = 1.0
        end

        if r < 4.0
            fhwR = 0.0
        elseif r < 8
            fhwR = a9 * (r - 4.0)/4.0
        elseif r < 18.0
            fhwR = a9
        elseif r < 24.0
            fhwR = a9 * (1.0 - (r - 18.0)/7.0)
        else
            fhwR = 0.0
        end

        f4 = fhwM * fhwR
    else
        f4 = 0.0
    end

    pga_rock = exp(f1 + f3 + f4)
    return pga_rock
end

	
"""
	as1997_base(idx::Int, pga_rock::Real, m::T, r::T, isSoil::Int, faultType::Int, hangingWall::Int=0) where T<:Real

Base model of Abrahamson & Silva (1997) where a period index is specified.

- `idx` the index associated with the period of interest
- `pga_rock` the level of peak ground acceleration on rock (units of 'g')
- `m` the magnitude
- `r` the rupture distance (units of 'km')
- `isSoil` equal to 1 for soil sites, and 0 for rock sites
- `faultType` equal to 2 for reverse, 1 for reverse/oblique, 0 otherwise
- `hangingWall` equal to 1 if site is on the hanging wall, 0 otherwise

Returns
- `μ_lnSa` the mean logarithmic spectral acceleration
- `σ_lnSa` the logarithmic standard deviation
"""
function as1997_base(idx::Int, pga_rock::Real, m::T, r::T, isSoil::Int, faultType::Int, hangingWall::Int=0) where T<:Real
    # get the coefficients for the given index
    c4, a1, a2, a3, a4, a5, a6, a9, a10, a11, a12, a13, c1, c5, n, b5, b6 = as1997_coefficients_for_index(idx)

    if m <= c1
        f1 = a1 + a2*(m - c1) + a12*(8.5 - m)^n + (a3 + a13*(m - c1))*log(sqrt(r^2 + c4^2))
    else
        f1 = a1 + a4*(m - c1) + a12*(8.5 - m)^n + (a3 + a13*(m - c1))*log(sqrt(r^2 + c4^2))
    end

    if faultType != 0
        if m <= 5.8
            f3 = a5
        elseif m < c1
            f3 = a5 + (a6 - a5)*(m - 5.8)/(c1 - 5.8)
        else
            f3 = a6
        end
        f3 *= faultType/2.0
    else
        f3 = 0.0
    end

    if hangingWall != 0
        if m <= 5.5
            fhwM = 0.0
        elseif m < 6.5
            fhwM = m - 5.5
        else
            fhwM = 1.0
        end

        if r < 4.0
            fhwR = 0.0
        elseif r < 8
            fhwR = a9 * (r - 4.0)/4.0
        elseif r < 18.0
            fhwR = a9
        elseif r < 24.0
            fhwR = a9 * (1.0 - (r - 18.0)/7.0)
        else
            fhwR = 0.0
        end

        f4 = fhwM * fhwR
    else
        f4 = 0.0
    end

    # compute the site response
    if isSoil > 0
        f5 = a10 + a11 * log( pga_rock + c5 )
    else
        f5 = 0.0
    end

    # get the median spectral acceleration
    μ_lnSa = f1 + f3 + f4 + f5

    # get the standard deviation
    if m <= 5.0
        σ_lnSa = b5
    elseif m >= 7.0
        σ_lnSa = b5 - 2b6
    else
        σ_lnSa = b5 - (m-5.0)*b6
    end

    return μ_lnSa, σ_lnSa
end


"""
	abrahamson_silva_1997(period::T, m::T, r::T, isSoil::Int, faultType::Int, hangingWall::Int=0) where T <: Real

Abrahamson & Silva (1997) empirical ground-motion model
Seismological Research Letters, Vol 68, No 1, p94

Input variables:  
- `period` = response period (s)
- `m`      = moment magnitude
- `r`      = rupture distance (km)
- `isSoil`  = 1 for soil prediction, 0 for rock
- `faultType` = 2 for reverse, 1 for reverse/oblique, 0 otherwise
- `hangingWall` = 1 for hanging wall sites, 0 otherwise

Output variables:
- `μ_lnSa`  = mean logarithmic spectral acceleration (Sa in 'g')
- `σ_lnSa`  = logarithmic standard deviation
"""
function abrahamson_silva_1997(period::T, m::T, r::T, isSoil::Int, faultType::Int, hangingWall::Int=0) where T<:Real

    # all soil site calculations require the pga on rock to be computed, so compute this up front if necessary
    if isSoil == 1
        pga_rock = as1997_pga_rock(m, r, faultType, hangingWall)
    else
        pga_rock = NaN
    end

    # for periods less than 0.01s, restrict to 0.01s
    if period < Ti[2]
        period = Ti[2]
    elseif period > Ti[end]
        period = Ti[end]
    end

    # check to see if interpolation is required
    id_lo = findlast(Ti .<= period)
    id_hi = findfirst(Ti .>= period)
    if id_lo == id_hi # no interpolation is required
        μ_lnSa, σ_lnSa = as1997_base(id_lo, pga_rock, m, r, isSoil, faultType, hangingWall)
    else
        μ_lnSa_lo, σ_lnSa_lo = as1997_base(id_lo, pga_rock, m, r, isSoil, faultType, hangingWall)
        μ_lnSa_hi, σ_lnSa_hi = as1997_base(id_hi, pga_rock, m, r, isSoil, faultType, hangingWall)
        # interpolate
        μ_lnSa = μ_lnSa_lo + log(period/Ti[id_lo])*(μ_lnSa_hi - μ_lnSa_lo)/log(Ti[id_hi]/Ti[id_lo])
        σ_lnSa = σ_lnSa_lo + log(period/Ti[id_lo])*(σ_lnSa_hi - σ_lnSa_lo)/log(Ti[id_hi]/Ti[id_lo])
    end
    return μ_lnSa, σ_lnSa
end

end

# ╔═╡ 1edf7652-7834-4919-aeed-58147a4d5980
begin
	# parameters of the GR source
	m_min = 5.0
	m_max = 8.0
	b_value = 1.0
	λm_min = 0.05
	grd = GutenbergRichter(m_min, m_max, b_value, λm_min)
end;

# ╔═╡ 784e5316-4297-49bc-b299-32a7c4735586
md"""
Define the properties of the GR source to be as follows:
- `m_min` = $(m_min)
- `m_max` = $(m_max)
- `b_value` = $(b_value)
- `λ_eq` = $(λm_min)
"""

# ╔═╡ 58bbbc09-4c78-4fff-bfcc-d460b3638cae
begin
	# number of magnitude bins
	Δm = 0.2
	num_m = Int((m_max - m_min) / Δm)
	mi = range(m_min+Δm/2, stop=m_max-Δm/2, step=Δm)
	
	# distance from site
	r_rup = 10.0

	# general arguments for AS1997 GMM (rock, strike-slip)
	is_soil = 0
	fault_type = 0
	hanging_wall = 0

	# create a standard normal distribution for the GMM probability calcs
	nd = Normal()
end;

# ╔═╡ c3bd8c1f-87b1-4ea2-8bf7-38b0087e137b
md"""
## Gutenberg-Richter Source
Consider the hazard for a GR source located at a fixed nominal distance from a rock site.

Define the distance between the source and the site to be a fixed distance of $r_{rup}=$ $(r_rup) km.
Note that this assumption is made for convenience, in reality it will be very uncommon for the rupture distance to not vary with magnittude.
However, for the purposes of this example, it is convenient to fix the distance and simply focus upon the role that the magnitude-frequency distribution plays on the hazard.
"""

# ╔═╡ ef8a081d-44de-41c3-8545-c56f988a7c09
md"""
The other parameters required for specifying the ground-motion predictions are:
- `r_rup` = $(r_rup)
- `is_soil` = $(is_soil)
- `fault_type` = $(fault_type)
- `hanging_wall` = $(hanging_wall)
"""

# ╔═╡ 160edc56-4280-4f1f-b71b-07731817575c
md"""
#### Discretisation of the magnitude range
We will work with discrete magnitude intervals, and can assess the impact of this discretisation.
The magnitude bins will have a width of $\Delta m$ = $(Δm).
"""

# ╔═╡ 602df306-16aa-4e9c-896d-a0df872e1ec2
begin
	λmi = zeros(num_m)
	Pmi = zeros(num_m)

	for (i, m) in enumerate(mi)
		λmi[i] = rate_of_occurrence(grd, m-Δm/2, m+Δm/2)
		Pmi[i] = cdf(grd, m+Δm/2) - cdf(grd, m-Δm/2)
	end
end


# ╔═╡ eadc4016-aab4-4aee-b989-5780aa1f61f0
@htl("""
The basic properties of the GR source can be viewed in the table below:
<table style="width: 70%; text-align: center">
	<caption>Magnitude-frequency results shown for a Gutenberg-Richter distribution with parameters: $(md"""``m_{\min}``: $(grd.m_min), ``m_{\max}``: $(grd.m_max), ``b``: $(grd.b_value) and ``\Delta m``: $(Δm)""") </caption>
	<colgroup>
       <col span="1" style="width: 33%;">
       <col span="1" style="width: 33%;">
       <col span="1" style="width: 33%;">
    </colgroup>
	<tbody>
		<tr style="border-bottom: 1px solid lightgray;"><th>$(md"""``m``""") <th>$(md"""``P(M=m)``""") <th>$(md"""``\lambda(M=m)``""")  </tr>
		$((@htl("<tr><td>$(m) <td>$(round(Pmi[i], sigdigits=3)) <td>$(round(λmi[i], sigdigits=3)) </tr>") for (i, m) in enumerate(mi) ))
		<tr style="border-top: 1px solid lightgray;"><td style="text-align: right;">Sum: <td>$(round(sum(Pmi), digits=3)) <td>$(round(sum(λmi), digits=3)) </tr>
	</tbody>
</table>
""")

# ╔═╡ a7bc4457-4d16-4457-8907-4f0d990400a4
begin
# add method for computing the expected magnitude within a bin
# this ensures that results are reasonable for the hazard calculations
# even if a coarse magnitude discretisation is adopted

"""
	expected_magnitude(dist::MagnitudeFrequencyDistribution, m_lo::T, m_hi::T) where T<:Real

Expected magnitude over a discrete magnitude range.
- `dist` is a concrete instance of a subtype of `MagnitudeFrequencyDistribution`
- `m_lo` is the lower magnitude of the interval
- `m_hi` is the upper magnitude of the interval
"""
function expected_magnitude(dist::MagnitudeFrequencyDistribution, m_lo::T, m_hi::T) where T<:Real
	numer = quadgk(m -> m * pdf(dist, m), m_lo, m_hi)[1]
	denom = cdf(dist, m_hi) - cdf(dist, m_lo)
	return numer / denom
end

μmi = zeros(num_m)
for i in 1:num_m
	μmi[i] = expected_magnitude(grd, mi[i]-Δm/2, mi[i]+Δm/2)
end
	
end;

# ╔═╡ f54e78ca-03af-4b58-8fb4-efd52bebb583
begin
	# number of Sa(T) values
	num_im = 11
	IMi = exp10.(range(-2, stop=0, length=num_im))

	num_ε = 7
	εi = range(-3, stop=3, length=num_ε)
	Δε = εi[2] - εi[1]
end;

# ╔═╡ fd501caf-52d0-41a2-ade1-81fcbeebcf67
md"""
## Hazard Curve
We can compute hazard curves for various periods:

Period:
	$(@bind periodID Slider(1:num_periods))
"""

# ╔═╡ f9c2f19a-29e4-47d8-a0b1-ae3686da794b
begin
	period = Ti[periodID]
	if period < 0.01
		imstr = "Peak ground acceleration, PGA [g]"
		λstr = "Annual rate of exceedance, λ(pga)"
		dstr = "PGA"
	else
		imstr = "Spectral acceleration, SA(T = $(period)s) [g]"
		λstr = "Annual rate of exceedance, λ(sa)"
		dstr = "SA(T = $(period)s)"
	end
end;

# ╔═╡ f1981dc0-cd80-4717-b811-bbb518d2e321
md"""
T = $(period) s
"""

# ╔═╡ 7b314136-36c4-4483-8479-6177740cebcd
begin
	λIMi = zeros(num_im)

	for i in 1:num_m
		μlnSa, σlnSa = abrahamson_silva_1997(period, μmi[i], r_rup, is_soil, fault_type)

		εIMi = ( log.(IMi) .- μlnSa ) ./ σlnSa
		PoEi = map(ε -> Distributions.ccdf(nd, ε), εIMi)

		λIM_Mi = PoEi * λmi[i]
		λIMi .+= λIM_Mi
	end

	plot(IMi, λIMi, xscale=:log10, yscale=:log10, marker=:circle, lab="")
	ylims!(exp10(floor(minimum(log10.(λIMi)))), exp10(ceil(log10(grd.λ_eq))))
	xlabel!(imstr)
	ylabel!(λstr)
end

# ╔═╡ 3d01024e-d6f9-4c81-aa0b-dd1e4e1bef4e
@htl("""
The hazard curve values can be viewed in the table below:
<table style="width: 70%; text-align: center">
	<caption>Hazard curve computed for a Gutenberg-Richter source distribution with parameters: $(md"""``m_{\min}``: $(grd.m_min), ``m_{\max}``: $(grd.m_max), ``b``: $(grd.b_value) and ``\Delta m``: $(Δm), at a distance ``r_{rup}``: $(r_rup) km.""") Results shown for $(imstr) </caption>
	<colgroup>
       <col span="1" style="width: 50%;">
       <col span="1" style="width: 50%;">
    </colgroup>
	<tbody>
		<tr style="border-bottom: 1px solid lightgray;"><th>$(md"""``im``""") <th>$(md"""``\lambda(IM>im)``""")  </tr>
		$((@htl("<tr><td>$(round(IMi[i], sigdigits=3)) <td>$(round(λIMi[i], sigdigits=3)) </tr>") for i in 1:num_im ))
	</tbody>
</table>
""")

# ╔═╡ 66ee4e00-daaf-476c-b8d6-b95a838fa086
md"""
## Disaggregation
Select a level of the intensity measure for which to display the disaggregation information:

IM level index: $(@bind imID Slider(1:num_im))
"""

# ╔═╡ 406afd70-acc0-42db-a477-8d0d2b0ea891
md"""
IM level selected: $(round(IMi[imID], digits=4)) [g] 
"""

# ╔═╡ 9c842512-52bb-4efb-9f97-44a90a020829
begin
	# disaggregation calculations here
	λMε = zeros(num_m, num_ε)
	lnIM_star = log(IMi[imID])
	
	# preparing the contributions for disaggregation plots
	for i in 1:num_m
		μlnSa, σlnSa = abrahamson_silva_1997(period, μmi[i], r_rup, is_soil, fault_type)
		ε_star = ( lnIM_star - μlnSa ) / σlnSa

		for j in 1:num_ε
			if j == 1
				εn = -Inf
			else
				εn = εi[j] - Δε/2
			end
			if j == num_ε
				εx = Inf
			else
				εx = εi[j] + Δε/2
			end
			if εn > ε_star
				pεj = Distributions.cdf(nd, εx) - Distributions.cdf(nd, εn)
			elseif εx > ε_star
				pεj = Distributions.cdf(nd, εx) - Distributions.cdf(nd, ε_star)
			else
				pεj = 0.0
			end
			λMε[i,j] = pεj * λmi[i]
		end
	end
end

# ╔═╡ d8745d60-4db4-4790-b12c-660db2946c0f
begin
	groupedbar(λMε[:,end:-1:1],
		bar_position=:stack,
		label = ["ε = +3" "ε = +2" "ε = +1" "ε = 0" "ε = -1" "ε = -2" "ε = -3" ],
		xticks=(1:num_m, string.(mi)),
		color = reshape(colormap("RdBu", num_ε), (1,num_ε)))
	ylabel!("Contribution to hazard")
	xlabel!("Magnitude")
	title!("Disaggregation for $(dstr) = $(round(IMi[imID], digits=4))g")
end

# ╔═╡ 91b29116-a6d0-4704-a14c-11ab30ac8f2f
begin
	pλMε = round.(λMε / λIMi[imID] * 100, digits=4)
	sum(pλMε)
end;

# ╔═╡ 6d1fbf41-1fd2-44fa-9f57-9c9f352a4bfa
@htl("""
The disaggregation values can be viewed in the table below. The values presented show the percentage contribution over the epsilon values for each level of the intensity measure. The sum over the final column should always equal 100%.

<table style="width: 100%; text-align: center">
	<caption>Disaggregation for $(dstr) = $(round(IMi[imID], sigdigits=4))g $(md"""``\lambda(IM>im)`` = $(round(λIMi[imID], sigdigits=4))""")</caption>
	<colgroup>
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
       <col span="1" style="width: 11%;">
    </colgroup>
	<tbody>
		<tr style="border-bottom: 1px solid lightgray;"> <th>$(md"""``m``""") <th>$(md"""``\varepsilon=-3``""") <th>$(md"""``\varepsilon=-2``""") <th>$(md"""``\varepsilon=-1``""") <th>$(md"""``\varepsilon= 0``""") <th>$(md"""``\varepsilon=+1``""") <th>$(md"""``\varepsilon=+2``""") <th>$(md"""``\varepsilon=+3``""") <th>$(md"""``\sum_i \lambda(\varepsilon_i)``""") </tr>
		$((@htl("<tr><td>$(round(mi[i], digits=1)) <td>$(pλMε[i,1]) <td>$(pλMε[i,2]) <td>$(pλMε[i,3]) <td>$(pλMε[i,4]) <td>$(pλMε[i,5]) <td>$(pλMε[i,6]) <td>$(pλMε[i,7]) <td>$(round(sum(λMε[i,:])/λIMi[imID]*100, digits=2)) </tr>") for i in 1:num_m ))
		<tr style="border-top: 1px solid lightgray;"><td> <td> <td> <td> <td> <td> <td> <td style="text-align: right;">Sum: <td>$(round(sum(pλMε), digits=2)) </tr>
	</tbody>
</table>
""")

# ╔═╡ 48054f83-9d81-42ce-bd64-96fd45def256
test_fun(2.0)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
ColorSchemes = "~3.15.0"
Distributions = "~0.25.24"
HypertextLiteral = "~0.9.2"
Plots = "~1.23.5"
PlutoUI = "~0.7.10"
QuadGK = "~2.4.2"
StatsPlots = "~0.14.33"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "72dcda9e19f88d09bf21b5f9507a0bb430bce2aa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.24"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "440165bf08bc500b8fe4a7be2dc83271a00c0716"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.12"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c8b8775b2f242c80ea85c83714c64ecfa3c53355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "7dc03c2b145168f5854085a16d054429d612b637"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.5"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "26b4d16873562469a0a1e6ae41d90dec9e51286d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.10"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "efd23b378ea5f2db53a55ae53d3133de4e080aa9"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.16"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "4d9c69d65f1b270ad092de0abe13e859b8c55cad"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.33"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═81bf285c-475d-466a-846f-82db6f6de9ab
# ╟─b32f2a7e-3dad-11ec-3854-773f1ca4cbbc
# ╟─87178931-1277-4c5a-b718-b15335a0622f
# ╟─5f514ffb-ad02-48b3-a2f4-1ea974562b03
# ╟─a6e100e1-39e8-4f2e-81b0-5b5d0982330c
# ╠═436c894f-7864-433f-86b8-6d3827290aa9
# ╟─891dc275-7869-4c20-99db-f7240363af4f
# ╟─7ea16336-3500-4214-8ba6-6b459da07f6c
# ╟─f8215680-b83c-4545-8e1d-c21f898a0eae
# ╟─e0dc1e81-037d-4828-830d-9085998cc356
# ╟─97e6a615-88c5-48aa-a8a7-b22221c8be91
# ╟─c890bc4c-3a20-4511-9e74-aa48a3a78e01
# ╟─58e3783d-c98b-4260-8e36-79d1c80fb7b2
# ╟─ba3c7dba-34dd-45a9-b8bf-bcc51d7fb625
# ╟─b570c8d4-c2f8-4dcb-b623-b1c1ea28026a
# ╟─bd36b64d-88ee-47ed-9509-5b05441abb2f
# ╟─346722a4-a1cb-4697-b093-defd57b3c610
# ╟─c4e39f09-1ebf-4eff-8a3e-cc9f81c2b288
# ╟─7be95d4b-ade1-48f6-8000-6e6e9409987e
# ╟─61db3c2a-d98b-4246-84c3-8731cb29e669
# ╟─313c5c6b-2eff-4df5-adae-a01c4eb8de9b
# ╟─dd3c5633-7295-454b-be67-ea38ded8850c
# ╟─0f1e9101-32ec-435e-ba2d-40d7215fe58a
# ╟─0fc11681-3776-4521-a67d-2aa7f5dce5fe
# ╟─c3bd8c1f-87b1-4ea2-8bf7-38b0087e137b
# ╟─1edf7652-7834-4919-aeed-58147a4d5980
# ╟─784e5316-4297-49bc-b299-32a7c4735586
# ╟─58bbbc09-4c78-4fff-bfcc-d460b3638cae
# ╟─ef8a081d-44de-41c3-8545-c56f988a7c09
# ╟─160edc56-4280-4f1f-b71b-07731817575c
# ╟─602df306-16aa-4e9c-896d-a0df872e1ec2
# ╟─eadc4016-aab4-4aee-b989-5780aa1f61f0
# ╟─a7bc4457-4d16-4457-8907-4f0d990400a4
# ╟─f54e78ca-03af-4b58-8fb4-efd52bebb583
# ╟─fd501caf-52d0-41a2-ade1-81fcbeebcf67
# ╟─f1981dc0-cd80-4717-b811-bbb518d2e321
# ╟─f9c2f19a-29e4-47d8-a0b1-ae3686da794b
# ╟─7b314136-36c4-4483-8479-6177740cebcd
# ╟─3d01024e-d6f9-4c81-aa0b-dd1e4e1bef4e
# ╟─66ee4e00-daaf-476c-b8d6-b95a838fa086
# ╟─406afd70-acc0-42db-a477-8d0d2b0ea891
# ╟─9c842512-52bb-4efb-9f97-44a90a020829
# ╟─d8745d60-4db4-4790-b12c-660db2946c0f
# ╟─91b29116-a6d0-4704-a14c-11ab30ac8f2f
# ╟─6d1fbf41-1fd2-44fa-9f57-9c9f352a4bfa
# ╠═afa78dcf-24b9-489e-a10b-907e2fb2452e
# ╠═48054f83-9d81-42ce-bd64-96fd45def256
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
