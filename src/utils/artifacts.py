import logging
import os
import json
import hashlib
import shutil
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import zipfile
import tarfile
import uuid
import warnings

logger = logging.getLogger(__name__)

# Cloud storage imports with error handling
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    logger.warning("AWS SDK not available. Install with: pip install boto3")
    AWS_AVAILABLE = False
    boto3 = ClientError = NoCredentialsError = None

try:
    from hdfs import InsecureClient, HdfsError
    HDFS_AVAILABLE = True
except ImportError:
    logger.warning("HDFS client not available. Install with: pip install hdfs")
    HDFS_AVAILABLE = False
    InsecureClient = HdfsError = None

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("Azure Storage SDK not available. Install with: pip install azure-storage-blob")
    AZURE_AVAILABLE = False
    BlobServiceClient = AzureError = None

try:
    from google.cloud import storage as gcs
    from google.cloud.exceptions import GoogleCloudError
    GCP_AVAILABLE = True
except ImportError:
    logger.warning("Google Cloud Storage SDK not available. Install with: pip install google-cloud-storage")
    GCP_AVAILABLE = False
    gcs = GoogleCloudError = None


class ArtifactError(Exception):
    pass


class StorageBackend:
    """Base class for storage backends."""
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        raise NotImplementedError
    
    def download_file(self, remote_path: str, local_path: str) -> str:
        raise NotImplementedError
    
    def list_files(self, prefix: str) -> List[str]:
        raise NotImplementedError
    
    def delete_file(self, remote_path: str) -> bool:
        raise NotImplementedError
    
    def file_exists(self, remote_path: str) -> bool:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        full_remote_path = self.base_path / remote_path
        full_remote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, full_remote_path)
        return str(full_remote_path)
    
    def download_file(self, remote_path: str, local_path: str) -> str:
        full_remote_path = self.base_path / remote_path
        if not full_remote_path.exists():
            raise ArtifactError(f"File not found: {full_remote_path}")
        shutil.copy2(full_remote_path, local_path)
        return local_path
    
    def list_files(self, prefix: str) -> List[str]:
        search_path = self.base_path / prefix
        if search_path.is_file():
            return [str(search_path.relative_to(self.base_path))]
        elif search_path.is_dir():
            return [str(p.relative_to(self.base_path)) for p in search_path.rglob('*') if p.is_file()]
        else:
            return []
    
    def delete_file(self, remote_path: str) -> bool:
        full_remote_path = self.base_path / remote_path
        if full_remote_path.exists():
            if full_remote_path.is_file():
                full_remote_path.unlink()
            else:
                shutil.rmtree(full_remote_path)
            return True
        return False
    
    def file_exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()


class S3StorageBackend(StorageBackend):
    """Amazon S3 storage backend."""
    
    def __init__(self, bucket_name: str, region_name: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        if not AWS_AVAILABLE:
            raise ArtifactError("AWS SDK not available. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.s3_client = session.client('s3')
        
        # Verify bucket access
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ArtifactError(f"S3 bucket not found: {bucket_name}")
            else:
                raise ArtifactError(f"S3 access error: {e}")
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, remote_path)
            return f"s3://{self.bucket_name}/{remote_path}"
        except ClientError as e:
            raise ArtifactError(f"S3 upload failed: {e}")
    
    def download_file(self, remote_path: str, local_path: str) -> str:
        try:
            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, remote_path, local_path)
            return local_path
        except ClientError as e:
            raise ArtifactError(f"S3 download failed: {e}")
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
            
            return files
        except ClientError as e:
            raise ArtifactError(f"S3 list failed: {e}")
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except ClientError as e:
            logger.warning(f"S3 delete failed: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except ClientError:
            return False


class HDFSStorageBackend(StorageBackend):
    """Hadoop HDFS storage backend."""
    
    def __init__(self, namenode_url: str, user: Optional[str] = None):
        if not HDFS_AVAILABLE:
            raise ArtifactError("HDFS client not available. Install with: pip install hdfs")
        
        try:
            self.client = InsecureClient(namenode_url, user=user)
            # Test connection
            self.client.status('/')
        except HdfsError as e:
            raise ArtifactError(f"HDFS connection failed: {e}")
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        try:
            # Ensure remote directory exists
            remote_dir = str(Path(remote_path).parent)
            if remote_dir != '.' and not self.client.status(remote_dir, strict=False):
                self.client.makedirs(remote_dir)
            
            self.client.upload(remote_path, local_path, overwrite=True)
            return remote_path
        except HdfsError as e:
            raise ArtifactError(f"HDFS upload failed: {e}")
    
    def download_file(self, remote_path: str, local_path: str) -> str:
        try:
            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download(remote_path, local_path, overwrite=True)
            return local_path
        except HdfsError as e:
            raise ArtifactError(f"HDFS download failed: {e}")
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            files = []
            if self.client.status(prefix, strict=False):
                for root, dirs, filenames in self.client.walk(prefix):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            return files
        except HdfsError as e:
            raise ArtifactError(f"HDFS list failed: {e}")
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            return self.client.delete(remote_path, recursive=True)
        except HdfsError as e:
            logger.warning(f"HDFS delete failed: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        try:
            return self.client.status(remote_path, strict=False) is not None
        except HdfsError:
            return False


class AzureStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, connection_string: str, container_name: str):
        if not AZURE_AVAILABLE:
            raise ArtifactError("Azure SDK not available. Install with: pip install azure-storage-blob")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_name = container_name
            
            # Ensure container exists
            try:
                self.blob_service_client.get_container_client(container_name).get_container_properties()
            except Exception:
                self.blob_service_client.create_container(container_name)
                
        except AzureError as e:
            raise ArtifactError(f"Azure storage initialization failed: {e}")
    
    def upload_file(self, local_path: str, remote_path: str) -> str:
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=remote_path
            )
            
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            return f"azure://{self.container_name}/{remote_path}"
        except AzureError as e:
            raise ArtifactError(f"Azure upload failed: {e}")
    
    def download_file(self, remote_path: str, local_path: str) -> str:
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=remote_path
            )
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            return local_path
        except AzureError as e:
            raise ArtifactError(f"Azure download failed: {e}")
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blob_list]
        except AzureError as e:
            raise ArtifactError(f"Azure list failed: {e}")
    
    def delete_file(self, remote_path: str) -> bool:
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=remote_path
            )
            blob_client.delete_blob()
            return True
        except AzureError as e:
            logger.warning(f"Azure delete failed: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=remote_path
            )
            return blob_client.exists()
        except AzureError:
            return False


class ArtifactManager:
    """
    Comprehensive artifact management system with support for multiple storage backends,
    versioning, metadata tracking, and automated artifact lifecycle management.
    """
    
    def __init__(self, storage_backend: StorageBackend,
                 project_name: str = "ml-pipeline",
                 enable_versioning: bool = True,
                 compression: str = "none",
                 metadata_format: str = "json"):
        """
        Initialize artifact manager.
        
        Args:
            storage_backend: Storage backend instance
            project_name: Name of the project for organizing artifacts
            enable_versioning: Whether to enable automatic versioning
            compression: Compression method ("none", "zip", "tar.gz")
            metadata_format: Format for metadata files ("json", "yaml")
        """
        self.storage_backend = storage_backend
        self.project_name = project_name
        self.enable_versioning = enable_versioning
        self.compression = compression
        self.metadata_format = metadata_format
        
        # Internal tracking
        self.artifacts_cache = {}
        self.version_history = {}
        
        logger.info(f"Initialized ArtifactManager for project: {project_name}")
    
    def save_artifact(self, artifact: Any, name: str, 
                     artifact_type: str = "object",
                     version: Optional[str] = None,
                     tags: Optional[Dict[str, str]] = None,
                     description: Optional[str] = None,
                     compress: Optional[bool] = None) -> Dict[str, Any]:
        """
        Save artifact with metadata and versioning.
        
        Args:
            artifact: Artifact object to save
            name: Name/identifier for the artifact
            artifact_type: Type of artifact ("model", "plot", "data", "report", "object")
            version: Specific version (auto-generated if None and versioning enabled)
            tags: Tags for the artifact
            description: Description of the artifact
            compress: Override compression setting
            
        Returns:
            Dictionary with artifact metadata
        """
        logger.info(f"Saving artifact: {name} (type: {artifact_type})")
        
        try:
            # Generate version if not provided
            if self.enable_versioning and version is None:
                version = self._generate_version(name)
            
            # Create artifact path
            artifact_path = self._build_artifact_path(name, artifact_type, version)
            
            # Save artifact to temporary file first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Determine file format and save
                if artifact_type == "model":
                    artifact_file = self._save_model(artifact, temp_path, name)
                elif artifact_type == "plot":
                    artifact_file = self._save_plot(artifact, temp_path, name)
                elif artifact_type == "data":
                    artifact_file = self._save_data(artifact, temp_path, name)
                elif artifact_type == "report":
                    artifact_file = self._save_report(artifact, temp_path, name)
                else:
                    artifact_file = self._save_object(artifact, temp_path, name)
                
                # Apply compression if requested
                use_compression = compress if compress is not None else (self.compression != "none")
                if use_compression:
                    artifact_file = self._compress_artifact(artifact_file, temp_path)
                
                # Calculate file hash for integrity
                file_hash = self._calculate_file_hash(artifact_file)
                
                # Create metadata
                metadata = self._create_metadata(
                    name=name,
                    artifact_type=artifact_type,
                    version=version,
                    tags=tags or {},
                    description=description,
                    file_hash=file_hash,
                    file_size=os.path.getsize(artifact_file),
                    compression=self.compression if use_compression else "none"
                )
                
                # Save metadata file
                metadata_file = self._save_metadata(metadata, temp_path, name)
                
                # Upload artifact and metadata to storage
                artifact_remote_path = f"{artifact_path}/{Path(artifact_file).name}"
                metadata_remote_path = f"{artifact_path}/metadata.{self.metadata_format}"
                
                artifact_uri = self.storage_backend.upload_file(str(artifact_file), artifact_remote_path)
                metadata_uri = self.storage_backend.upload_file(str(metadata_file), metadata_remote_path)
                
                # Update local cache and version history
                self.artifacts_cache[name] = metadata
                if name not in self.version_history:
                    self.version_history[name] = []
                self.version_history[name].append(version)
                
                # Update metadata with storage URIs
                metadata.update({
                    'artifact_uri': artifact_uri,
                    'metadata_uri': metadata_uri,
                    'remote_path': artifact_remote_path
                })
                
                logger.info(f"Successfully saved artifact: {name} (version: {version})")
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to save artifact {name}: {e}")
            raise ArtifactError(f"Artifact save failed: {e}")
    
    def load_artifact(self, name: str, version: Optional[str] = None,
                     cache_locally: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Load artifact by name and version.
        
        Args:
            name: Artifact name
            version: Specific version (latest if None)
            cache_locally: Whether to cache the artifact locally
            
        Returns:
            Tuple of (artifact_object, metadata)
        """
        logger.info(f"Loading artifact: {name} (version: {version or 'latest'})")
        
        try:
            # Get artifact metadata
            metadata = self.get_artifact_metadata(name, version)
            
            # Determine artifact path
            artifact_path = self._build_artifact_path(
                name, metadata['artifact_type'], metadata['version']
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download artifact
                artifact_remote_path = metadata['remote_path']
                local_artifact_path = temp_path / Path(artifact_remote_path).name
                
                self.storage_backend.download_file(artifact_remote_path, str(local_artifact_path))
                
                # Verify file integrity
                if not self._verify_file_integrity(local_artifact_path, metadata['file_hash']):
                    raise ArtifactError("Artifact integrity check failed")
                
                # Decompress if needed
                if metadata.get('compression', 'none') != 'none':
                    local_artifact_path = self._decompress_artifact(local_artifact_path, temp_path)
                
                # Load artifact based on type
                artifact_type = metadata['artifact_type']
                
                if artifact_type == "model":
                    artifact = self._load_model(local_artifact_path)
                elif artifact_type == "plot":
                    artifact = self._load_plot(local_artifact_path)
                elif artifact_type == "data":
                    artifact = self._load_data(local_artifact_path)
                elif artifact_type == "report":
                    artifact = self._load_report(local_artifact_path)
                else:
                    artifact = self._load_object(local_artifact_path)
                
                logger.info(f"Successfully loaded artifact: {name}")
                return artifact, metadata
                
        except Exception as e:
            logger.error(f"Failed to load artifact {name}: {e}")
            raise ArtifactError(f"Artifact load failed: {e}")
    
    def list_artifacts(self, artifact_type: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None,
                      name_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List artifacts with optional filtering.
        
        Args:
            artifact_type: Filter by artifact type
            tags: Filter by tags
            name_pattern: Filter by name pattern (simple string matching)
            
        Returns:
            List of artifact metadata dictionaries
        """
        logger.info("Listing artifacts")
        
        try:
            artifacts = []
            
            # List all artifacts in storage
            prefix = f"{self.project_name}/artifacts/"
            remote_files = self.storage_backend.list_files(prefix)
            
            # Extract unique artifact names and versions
            artifact_info = {}
            for file_path in remote_files:
                if file_path.endswith(f"metadata.{self.metadata_format}"):
                    # Parse artifact path
                    path_parts = Path(file_path).parts
                    if len(path_parts) >= 4:  # project/artifacts/type_name_version/metadata.json
                        artifact_id = path_parts[-2]  # type_name_version
                        
                        # Download and parse metadata
                        with tempfile.NamedTemporaryFile() as temp_file:
                            self.storage_backend.download_file(file_path, temp_file.name)
                            
                            with open(temp_file.name, 'r') as f:
                                if self.metadata_format == "json":
                                    metadata = json.load(f)
                                else:
                                    import yaml
                                    metadata = yaml.safe_load(f)
                            
                            # Apply filters
                            if artifact_type and metadata.get('artifact_type') != artifact_type:
                                continue
                            
                            if name_pattern and name_pattern not in metadata.get('name', ''):
                                continue
                            
                            if tags:
                                artifact_tags = metadata.get('tags', {})
                                if not all(artifact_tags.get(k) == v for k, v in tags.items()):
                                    continue
                            
                            artifacts.append(metadata)
            
            logger.info(f"Found {len(artifacts)} artifacts")
            return artifacts
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            raise ArtifactError(f"Artifact listing failed: {e}")
    
    def get_artifact_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a specific artifact.
        
        Args:
            name: Artifact name
            version: Specific version (latest if None)
            
        Returns:
            Artifact metadata dictionary
        """
        try:
            # Check local cache first
            if name in self.artifacts_cache and version is None:
                return self.artifacts_cache[name]
            
            # Find artifact in storage
            if version is None:
                # Get latest version
                versions = self.get_artifact_versions(name)
                if not versions:
                    raise ArtifactError(f"Artifact not found: {name}")
                version = versions[-1]  # Assuming versions are sorted
            
            # Build metadata path
            artifact_path = self._build_artifact_path(name, "", version)  # Type will be in metadata
            metadata_path = f"{artifact_path}/metadata.{self.metadata_format}"
            
            # Download and parse metadata
            with tempfile.NamedTemporaryFile() as temp_file:
                self.storage_backend.download_file(metadata_path, temp_file.name)
                
                with open(temp_file.name, 'r') as f:
                    if self.metadata_format == "json":
                        metadata = json.load(f)
                    else:
                        import yaml
                        metadata = yaml.safe_load(f)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to get metadata for {name}: {e}")
            raise ArtifactError(f"Metadata retrieval failed: {e}")
    
    def get_artifact_versions(self, name: str) -> List[str]:
        """
        Get all versions of an artifact.
        
        Args:
            name: Artifact name
            
        Returns:
            List of version strings
        """
        try:
            versions = []
            
            # Search for artifacts with this name
            prefix = f"{self.project_name}/artifacts/"
            remote_files = self.storage_backend.list_files(prefix)
            
            for file_path in remote_files:
                if f"/{name}_" in file_path and file_path.endswith(f"metadata.{self.metadata_format}"):
                    # Extract version from path
                    path_parts = Path(file_path).parts
                    if len(path_parts) >= 4:
                        artifact_id = path_parts[-2]  # type_name_version
                        
                        # Parse version from artifact_id
                        if artifact_id.startswith(f"object_{name}_"):
                            version = artifact_id[len(f"object_{name}_"):]
                            versions.append(version)
                        else:
                            # Try other artifact types
                            for atype in ["model", "plot", "data", "report"]:
                                prefix_pattern = f"{atype}_{name}_"
                                if artifact_id.startswith(prefix_pattern):
                                    version = artifact_id[len(prefix_pattern):]
                                    versions.append(version)
                                    break
            
            # Sort versions (simple string sort, could be improved with semantic versioning)
            versions.sort()
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get versions for {name}: {e}")
            raise ArtifactError(f"Version retrieval failed: {e}")
    
    def delete_artifact(self, name: str, version: Optional[str] = None,
                       delete_all_versions: bool = False) -> bool:
        """
        Delete artifact(s).
        
        Args:
            name: Artifact name
            version: Specific version (all versions if None and delete_all_versions=True)
            delete_all_versions: Whether to delete all versions
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting artifact: {name} (version: {version}, all: {delete_all_versions})")
        
        try:
            if delete_all_versions:
                versions = self.get_artifact_versions(name)
                success = True
                for v in versions:
                    try:
                        self._delete_single_artifact(name, v)
                    except Exception as e:
                        logger.warning(f"Failed to delete version {v}: {e}")
                        success = False
                return success
            else:
                if version is None:
                    # Delete latest version
                    versions = self.get_artifact_versions(name)
                    if versions:
                        version = versions[-1]
                    else:
                        raise ArtifactError(f"No versions found for {name}")
                
                return self._delete_single_artifact(name, version)
                
        except Exception as e:
            logger.error(f"Failed to delete artifact {name}: {e}")
            raise ArtifactError(f"Artifact deletion failed: {e}")
    
    def _delete_single_artifact(self, name: str, version: str) -> bool:
        """Delete a single artifact version."""
        try:
            # Get metadata to determine artifact type
            metadata = self.get_artifact_metadata(name, version)
            
            # Build artifact path
            artifact_path = self._build_artifact_path(name, metadata['artifact_type'], version)
            
            # Delete all files in artifact directory
            prefix = f"{artifact_path}/"
            files = self.storage_backend.list_files(prefix)
            
            success = True
            for file_path in files:
                if not self.storage_backend.delete_file(file_path):
                    success = False
            
            # Remove from local cache
            if name in self.artifacts_cache:
                del self.artifacts_cache[name]
            
            # Remove from version history
            if name in self.version_history:
                try:
                    self.version_history[name].remove(version)
                    if not self.version_history[name]:
                        del self.version_history[name]
                except ValueError:
                    pass
            
            return success
            
        except Exception as e:
            logger.warning(f"Failed to delete single artifact {name}:{version}: {e}")
            return False
    
    def create_artifact_bundle(self, artifact_names: List[str], 
                             bundle_name: str,
                             description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a bundle containing multiple artifacts.
        
        Args:
            artifact_names: List of artifact names to include
            bundle_name: Name for the bundle
            description: Description of the bundle
            
        Returns:
            Bundle metadata
        """
        logger.info(f"Creating artifact bundle: {bundle_name}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                bundle_dir = temp_path / bundle_name
                bundle_dir.mkdir()
                
                bundle_manifest = {
                    'bundle_name': bundle_name,
                    'description': description,
                    'created_at': datetime.now().isoformat(),
                    'artifacts': {}
                }
                
                # Download and package each artifact
                for artifact_name in artifact_names:
                    try:
                        metadata = self.get_artifact_metadata(artifact_name)
                        
                        # Download artifact
                        artifact_dir = bundle_dir / artifact_name
                        artifact_dir.mkdir()
                        
                        remote_path = metadata['remote_path']
                        local_path = artifact_dir / Path(remote_path).name
                        
                        self.storage_backend.download_file(remote_path, str(local_path))
                        
                        # Add to manifest
                        bundle_manifest['artifacts'][artifact_name] = {
                            'version': metadata['version'],
                            'artifact_type': metadata['artifact_type'],
                            'file_name': Path(remote_path).name,
                            'original_metadata': metadata
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to include artifact {artifact_name} in bundle: {e}")
                
                # Save bundle manifest
                manifest_path = bundle_dir / "bundle_manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump(bundle_manifest, f, indent=2, default=str)
                
                # Create bundle archive
                bundle_archive = temp_path / f"{bundle_name}.tar.gz"
                with tarfile.open(bundle_archive, 'w:gz') as tar:
                    tar.add(bundle_dir, arcname=bundle_name)
                
                # Save bundle as artifact
                bundle_metadata = self.save_artifact(
                    artifact=bundle_archive,
                    name=bundle_name,
                    artifact_type="bundle",
                    description=f"Artifact bundle: {description or bundle_name}"
                )
                
                logger.info(f"Created bundle {bundle_name} with {len(bundle_manifest['artifacts'])} artifacts")
                return bundle_metadata
                
        except Exception as e:
            logger.error(f"Failed to create bundle: {e}")
            raise ArtifactError(f"Bundle creation failed: {e}")
    
    def extract_artifact_bundle(self, bundle_name: str, 
                              target_directory: Optional[str] = None) -> List[str]:
        """
        Extract an artifact bundle.
        
        Args:
            bundle_name: Name of the bundle to extract
            target_directory: Directory to extract to (temp dir if None)
            
        Returns:
            List of extracted artifact names
        """
        logger.info(f"Extracting artifact bundle: {bundle_name}")
        
        try:
            # Load bundle
            bundle_data, bundle_metadata = self.load_artifact(bundle_name)
            
            if target_directory is None:
                target_directory = tempfile.mkdtemp()
            
            extract_path = Path(target_directory)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            # Extract bundle archive
            if isinstance(bundle_data, Path) and tarfile.is_tarfile(bundle_data):
                with tarfile.open(bundle_data, 'r:gz') as tar:
                    tar.extractall(extract_path)
            else:
                raise ArtifactError("Bundle is not a valid tar.gz file")
            
            # Read bundle manifest
            manifest_path = extract_path / bundle_name / "bundle_manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            extracted_artifacts = list(manifest['artifacts'].keys())
            logger.info(f"Extracted {len(extracted_artifacts)} artifacts from bundle")
            
            return extracted_artifacts
            
        except Exception as e:
            logger.error(f"Failed to extract bundle: {e}")
            raise ArtifactError(f"Bundle extraction failed: {e}")
    
    def _generate_version(self, name: str) -> str:
        """Generate version string for artifact."""
        if name in self.version_history and self.version_history[name]:
            # Increment version number
            last_version = self.version_history[name][-1]
            try:
                # Try to parse as semantic version
                parts = last_version.split('.')
                if len(parts) >= 3:
                    patch = int(parts[2]) + 1
                    return f"{parts[0]}.{parts[1]}.{patch}"
                else:
                    # Simple integer increment
                    return str(int(last_version) + 1)
            except ValueError:
                # Fallback to timestamp
                return datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # First version
            return "1.0.0"
    
    def _build_artifact_path(self, name: str, artifact_type: str, version: Optional[str]) -> str:
        """Build artifact storage path."""
        if version:
            artifact_id = f"{artifact_type}_{name}_{version}"
        else:
            artifact_id = f"{artifact_type}_{name}"
        
        return f"{self.project_name}/artifacts/{artifact_id}"
    
    def _save_model(self, model: Any, temp_path: Path, name: str) -> Path:
        """Save model artifact."""
        model_path = temp_path / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path
    
    def _save_plot(self, plot: Any, temp_path: Path, name: str) -> Path:
        """Save plot artifact."""
        import matplotlib.pyplot as plt
        
        plot_path = temp_path / f"{name}_plot.png"
        
        if hasattr(plot, 'savefig'):  # Matplotlib figure
            plot.savefig(plot_path, dpi=300, bbox_inches='tight')
        elif isinstance(plot, (str, Path)):  # File path
            shutil.copy2(plot, plot_path)
        else:
            raise ArtifactError(f"Unsupported plot type: {type(plot)}")
        
        return plot_path
    
    def _save_data(self, data: Any, temp_path: Path, name: str) -> Path:
        """Save data artifact."""
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            data_path = temp_path / f"{name}_data.parquet"
            data.to_parquet(data_path, index=False)
        elif isinstance(data, np.ndarray):
            data_path = temp_path / f"{name}_data.npy"
            np.save(data_path, data)
        elif isinstance(data, (str, Path)):  # File path
            data_path = temp_path / f"{name}_data{Path(data).suffix}"
            shutil.copy2(data, data_path)
        else:
            # Fallback to pickle
            data_path = temp_path / f"{name}_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
        
        return data_path
    
    def _save_report(self, report: Any, temp_path: Path, name: str) -> Path:
        """Save report artifact."""
        if isinstance(report, str):
            # Text report
            report_path = temp_path / f"{name}_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
        elif isinstance(report, (str, Path)):  # File path
            report_path = temp_path / f"{name}_report{Path(report).suffix}"
            shutil.copy2(report, report_path)
        else:
            # JSON serializable object
            report_path = temp_path / f"{name}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report_path
    
    def _save_object(self, obj: Any, temp_path: Path, name: str) -> Path:
        """Save generic object artifact."""
        obj_path = temp_path / f"{name}_object.pkl"
        with open(obj_path, 'wb') as f:
            pickle.dump(obj, f)
        return obj_path
    
    def _load_model(self, file_path: Path) -> Any:
        """Load model artifact."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_plot(self, file_path: Path) -> Path:
        """Load plot artifact (return path)."""
        return file_path
    
    def _load_data(self, file_path: Path) -> Any:
        """Load data artifact."""
        import pandas as pd
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix == '.npy':
            return np.load(file_path)
        elif suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Return file path for unknown formats
            return file_path
    
    def _load_report(self, file_path: Path) -> Any:
        """Load report artifact."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r') as f:
                return f.read()
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            # Return file path for other formats
            return file_path
    
    def _load_object(self, file_path: Path) -> Any:
        """Load generic object artifact."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _compress_artifact(self, file_path: Path, temp_path: Path) -> Path:
        """Compress artifact file."""
        if self.compression == "zip":
            compressed_path = temp_path / f"{file_path.stem}.zip"
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(file_path, file_path.name)
        elif self.compression == "tar.gz":
            compressed_path = temp_path / f"{file_path.stem}.tar.gz"
            with tarfile.open(compressed_path, 'w:gz') as tf:
                tf.add(file_path, file_path.name)
        else:
            return file_path
        
        return compressed_path
    
    def _decompress_artifact(self, file_path: Path, temp_path: Path) -> Path:
        """Decompress artifact file."""
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(temp_path)
                # Return path to extracted file
                extracted_files = zf.namelist()
                if extracted_files:
                    return temp_path / extracted_files[0]
        elif file_path.suffixes == ['.tar', '.gz']:
            with tarfile.open(file_path, 'r:gz') as tf:
                tf.extractall(temp_path)
                # Return path to extracted file
                extracted_files = tf.getnames()
                if extracted_files:
                    return temp_path / extracted_files[0]
        
        return file_path
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using hash."""
        actual_hash = self._calculate_file_hash(file_path)
        return actual_hash == expected_hash
    
    def _create_metadata(self, name: str, artifact_type: str, version: Optional[str],
                        tags: Dict[str, str], description: Optional[str],
                        file_hash: str, file_size: int, compression: str) -> Dict[str, Any]:
        """Create artifact metadata."""
        return {
            'name': name,
            'artifact_type': artifact_type,
            'version': version,
            'description': description,
            'tags': tags,
            'file_hash': file_hash,
            'file_size': file_size,
            'compression': compression,
            'created_at': datetime.now().isoformat(),
            'project_name': self.project_name,
            'framework': 'ml-pipeline-framework'
        }
    
    def _save_metadata(self, metadata: Dict[str, Any], temp_path: Path, name: str) -> Path:
        """Save metadata to file."""
        metadata_path = temp_path / f"{name}_metadata.{self.metadata_format}"
        
        with open(metadata_path, 'w') as f:
            if self.metadata_format == "json":
                json.dump(metadata, f, indent=2, default=str)
            else:
                import yaml
                yaml.dump(metadata, f, default_flow_style=False)
        
        return metadata_path
    
    @classmethod
    def create_local_backend(cls, base_path: str) -> 'ArtifactManager':
        """Create artifact manager with local storage backend."""
        backend = LocalStorageBackend(base_path)
        return cls(storage_backend=backend)
    
    @classmethod
    def create_s3_backend(cls, bucket_name: str, region_name: Optional[str] = None,
                         aws_access_key_id: Optional[str] = None,
                         aws_secret_access_key: Optional[str] = None) -> 'ArtifactManager':
        """Create artifact manager with S3 storage backend."""
        backend = S3StorageBackend(bucket_name, region_name, aws_access_key_id, aws_secret_access_key)
        return cls(storage_backend=backend)
    
    @classmethod
    def create_hdfs_backend(cls, namenode_url: str, user: Optional[str] = None) -> 'ArtifactManager':
        """Create artifact manager with HDFS storage backend."""
        backend = HDFSStorageBackend(namenode_url, user)
        return cls(storage_backend=backend)
    
    @classmethod
    def create_azure_backend(cls, connection_string: str, container_name: str) -> 'ArtifactManager':
        """Create artifact manager with Azure storage backend."""
        backend = AzureStorageBackend(connection_string, container_name)
        return cls(storage_backend=backend)