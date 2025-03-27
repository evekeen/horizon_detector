#!/bin/bash

# Set up AWS authentication
if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Setting up AWS authentication..."
    mkdir -p ~/.aws
    cat > ~/.aws/credentials << EOL
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
EOL

    if [ ! -z "$AWS_DEFAULT_REGION" ]; then
        cat > ~/.aws/config << EOL
[default]
region = ${AWS_DEFAULT_REGION}
EOL
    fi
fi