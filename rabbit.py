from theoden.security.auth import AuthenticationManager, UserRole

AuthenticationManager.create_yaml_and_users(
    "users.yaml", "topology.yaml", create_users=False
)
