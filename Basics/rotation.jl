### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 18fdb1ae-14e2-11eb-31ec-61303c211cbb
begin
	using LinearAlgebra
	using StaticArrays
	using Random
	using Distributions
end

# ╔═╡ f5aa34b0-167c-11eb-3c92-cfd242732c90
md"""
# This is a note for 3D Coordinates and Representations of Rotations (Cyrill Stachniss, 2020):
https://www.youtube.com/watch?v=YXGUGSAv09A&list=PLgnQpQtFTOGQ7eZU0tzmyjSV5w5lt28p8&index=4
"""

# ╔═╡ e06143ec-1171-11eb-0c7b-6f679044b95a
struct Point{T}
	coord::AbstractVector{T}
end

# ╔═╡ 9e62c0d4-128d-11eb-3954-1175756cb87d
struct EulerAngle
	θ::AbstractVector
end

# ╔═╡ 4de2b1b8-128e-11eb-08f7-ef7b65cf1517
struct AxisAngle
	r::AbstractVector
end

# ╔═╡ 3a73228a-14e0-11eb-0a4d-bbf949a0316b
md"##### Two properties of rotation matrix: det(r) == 1; inv(r) == transpose(r)"

# ╔═╡ 7575ff78-163a-11eb-0285-0788b2e5d43b
md"##### The order of yaw, pitch, and roll matters"

# ╔═╡ 32a8b1d6-163a-11eb-0ed0-696834bdb3e5
md"##### θ=0 is a singular point for transforming between axis angle and rotation matrix"

# ╔═╡ 7393bc30-1640-11eb-3811-89593e6ce6f8
md"""##### cosβ=0 i.e. β=+-90 is a singularity ("Glimbal Lock")"""

# ╔═╡ b3dc7562-163f-11eb-3b5d-dbba1933d002
md"##### Euler angles are unique given a rotation matrix, if the angle is in range -π to π. The rotation matrix is unique given euler angles"

# ╔═╡ 784fa84e-1675-11eb-3591-f7ad857e2638
md"""
##### Both Euler Angles and Axis angle has singularities and discountinuities (2π and 0 are the same, this may screw up the gradients).
##### Similar to Rotation Matrix, Quaternion has no singularities or discountinuities. Plus Quaternion only has 4 parameters compared to 9 parameters in Rotation Matrix. Quaternions are almost minimal
"""

# ╔═╡ 881043b4-1626-11eb-156f-4535bcb9baa7
struct Quaternion{T<:Number}
	qr::T
	qi::AbstractVector{T}
end

# ╔═╡ 060406a2-1677-11eb-3b56-3ff06359004d
begin
	import Base: +,*
	import LinearAlgebra.inv
	function +(q::Quaternion, r::Quaternion)::Quaternion
		Quaternion(q.qr+r.qr, q.qi+r.qi)
	end
	function *(q::Quaternion, r::Quaternion)::Quaternion #compose two rotation
		pv = r.qr.*q.qi + q.qr*r.qi + cross(q.qi, r.qi)
		Quaternion(q.qr*r.qr-dot(q.qi,r.qi), pv)
	end
	function inv(q::Quaternion)
		Quaternion(q.qr, -q.qi)
	end
end

# ╔═╡ c0809274-14de-11eb-177a-df5959f8fdd6
function assert_rotation_matrix(m::AbstractArray)
	@assert det(m) ≈ 1 "det is $(det(m))"
	@assert transpose(m) * m ≈ I "inv is $(inv(m)) and transpose is $(transpose(m))"
	#@assert all(norm(m, 1) .≈ 1) # this can be proved from the 2nd property
end

# ╔═╡ 27ba5326-11ba-11eb-00ad-8d7529740c52
struct Rotation
	matrix::AbstractMatrix
	function Rotation(m::AbstractMatrix)
		assert_rotation_matrix(m)
		new(m)
	end
end

# ╔═╡ 06fbad32-14e0-11eb-19f1-6ba91539f5fd
function to_rotation(θ::Number)::Rotation
	return Rotation([cos(θ) -sin(θ); sin(θ) cos(θ)])
end

# ╔═╡ 107f0ef4-14df-11eb-209a-65cafdd5cc7f
#Euler Angles to Rotation Matrix
function to_rotation(angle::EulerAngle)::Rotation
	rx = [1 0 0;
		0 cos(angle.θ[1]) -sin(angle.θ[1]);
		0 sin(angle.θ[1]) cos(angle.θ[1])]
	ry = [cos(angle.θ[2]) 0 sin(angle.θ[2]);
		0 1 0;
		-sin(angle.θ[2]) 0 cos(angle.θ[2])]
	rz = [cos(angle.θ[3]) -sin(angle.θ[3]) 0;
		sin(angle.θ[3]) cos(angle.θ[3]) 0;
		0 0 1]
	return Rotation(rz*ry*rx) #Yaw, Pitch, Roll
end

# ╔═╡ 456fa05e-14e1-11eb-2bec-17daecca21f4
#Axis Angle to Rotation Matrix
function to_rotation(a::AxisAngle)::Rotation
	θ = norm(a.r)
	if θ == 0
		return Rotation(Matrix(I,3,3)) #singularity #1
	end
	Sθ = [0 -a.r[3] a.r[2]; a.r[3] 0 -a.r[1]; -a.r[2] a.r[1] 0]
	Sr = Sθ / θ
	S_r = I + sin(θ)*Sr + (1-cos(θ))*(Sr)^2
	Rotation(S_r)
end

# ╔═╡ 753e5bd0-163b-11eb-12fd-059fd1243f19
#Rotation Matrix to Euler angles
function to_euler_angle(r::Rotation)::EulerAngle
	if r.matrix[3,2] == 0 && r.matrix[3,3] == 0
		β = asin(-r.matrix[3,1])
		α = 0
		if r.matrix[3,1] < 0 #singularity #2
			γ = atan(-r.matrix[1,2], r.matrix[1,3])
		else
			γ = atan(-r.matrix[1,2], -r.matrix[1,3])
		end
	else
		α=atan(r.matrix[3,2], r.matrix[3,3])
		β=atan(-r.matrix[3,1], √(r.matrix[3,2]^2+r.matrix[3,3]^2))
		γ=atan(r.matrix[2,1], r.matrix[1,1])
	end
	return EulerAngle([α, β, γ])
end

# ╔═╡ 680cffa6-11b1-11eb-340c-854809b66558
begin
	function assert_euler(θ::AbstractVector)
		θs = EulerAngle(θ)
		rot = to_rotation(θs)
		θs_back = to_euler_angle(rot)
		rot_again = to_rotation(θs_back)
		@assert rot.matrix ≈ rot_again.matrix
		#@assert θs.θ ≈ θs_back.θ  "$θs, $(θs_back.θ)" #Euler angles are unique if the input is in range -π/2 to π/2 and β != +-90
	end
	assert_euler(rand(Uniform(-π/2,π/2), (3)))
	assert_euler([2, π/2, 1])
end

# ╔═╡ 170c1c16-1634-11eb-0723-c96d2ee0f605
#Rotation Matrix to Axis Angle
function to_axis_angle(r::Rotation)::AxisAngle
	a = - [r.matrix[2,3]-r.matrix[3,2], r.matrix[3,1]-r.matrix[1,3], r.matrix[1,2]-r.matrix[2,1]]
	a_norm = norm(a)
	r_trace = tr(r.matrix)
	θ = atan(a_norm, r_trace-1)
	if a_norm == 0 #singularity #1
		return AxisAngle(a)
	end
	AxisAngle(θ*a/a_norm)
end

# ╔═╡ 86fbb332-163f-11eb-25f3-6ddfd26bb1e3
begin
	function assert_axis_angle(v::AbstractVector)
		ro = to_rotation(AxisAngle(v))
		axisangle = to_axis_angle(ro)
		ro_again = to_rotation(axisangle)
		@assert ro.matrix ≈ ro_again.matrix
		@assert v ≈ axisangle.r "$v, $axisangle"
	end
	θ = rand(Uniform(-π,π), (1))
	v = rand(Uniform(0,1), (3))
	assert_axis_angle(θ .* v / norm(v))
	assert_axis_angle([π,0,0])
end

# ╔═╡ 454933d2-167a-11eb-0a0c-13d4c0a2ee34
function to_quaternion(a::AxisAngle)::Quaternion
	θ = norm(a.r)
	r = a.r/θ
	Quaternion(cos(θ/2), sin(θ/2).*r)
end

# ╔═╡ 82ec8004-1172-11eb-2a04-b516d8620b1a
begin
	function rotate(p::Point, r::Rotation)::Point
		return Point(r.matrix * p.coord)
	end
	function rotate(p::Point, q::Quaternion)::Point
		pq = Quaternion(0.0, p.coord)
		p_result = q * pq * inv(q)
		#@assert p_result.qr ≈ 0 "$p_result" #qi is very small, but may not pass the approx
		return Point(p_result.qi)
	end
	function rotate(p::Point, r::AxisAngle)::Point
		return rotate(p, to_rotation(r))
	end
end

# ╔═╡ 16ee7448-167c-11eb-3190-b97209d23aae
begin
	θ1 = rand(Uniform(-π,π), (1))
	v1 = rand(Uniform(0,1), (3))
	a = AxisAngle(θ1 .* v1 / norm(v1))
	p = Point(rand(Uniform(0,1), (3)))
	p1 = rotate(p, a)
	p2 = rotate(p, to_quaternion(a))
	@assert p1.coord ≈ p2.coord "$p1, $p2"
end

# ╔═╡ 8cd80f70-1188-11eb-0b0c-7518741ac78d
function scale(p::Point, λ::Number)
	return Point(p.coord.+λ)
end

# ╔═╡ Cell order:
# ╟─f5aa34b0-167c-11eb-3c92-cfd242732c90
# ╠═18fdb1ae-14e2-11eb-31ec-61303c211cbb
# ╠═e06143ec-1171-11eb-0c7b-6f679044b95a
# ╠═9e62c0d4-128d-11eb-3954-1175756cb87d
# ╠═4de2b1b8-128e-11eb-08f7-ef7b65cf1517
# ╟─3a73228a-14e0-11eb-0a4d-bbf949a0316b
# ╠═c0809274-14de-11eb-177a-df5959f8fdd6
# ╠═27ba5326-11ba-11eb-00ad-8d7529740c52
# ╠═06fbad32-14e0-11eb-19f1-6ba91539f5fd
# ╟─7575ff78-163a-11eb-0285-0788b2e5d43b
# ╠═107f0ef4-14df-11eb-209a-65cafdd5cc7f
# ╟─32a8b1d6-163a-11eb-0ed0-696834bdb3e5
# ╠═456fa05e-14e1-11eb-2bec-17daecca21f4
# ╟─7393bc30-1640-11eb-3811-89593e6ce6f8
# ╠═753e5bd0-163b-11eb-12fd-059fd1243f19
# ╠═680cffa6-11b1-11eb-340c-854809b66558
# ╠═170c1c16-1634-11eb-0723-c96d2ee0f605
# ╟─b3dc7562-163f-11eb-3b5d-dbba1933d002
# ╠═86fbb332-163f-11eb-25f3-6ddfd26bb1e3
# ╟─784fa84e-1675-11eb-3591-f7ad857e2638
# ╠═881043b4-1626-11eb-156f-4535bcb9baa7
# ╠═060406a2-1677-11eb-3b56-3ff06359004d
# ╠═454933d2-167a-11eb-0a0c-13d4c0a2ee34
# ╠═82ec8004-1172-11eb-2a04-b516d8620b1a
# ╠═16ee7448-167c-11eb-3190-b97209d23aae
# ╠═8cd80f70-1188-11eb-0b0c-7518741ac78d
