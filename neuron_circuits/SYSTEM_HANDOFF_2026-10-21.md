# System handoff — 2026-10-21

## Immediate status

**Do not reboot into Ubuntu yet.** The running Ubuntu session is healthy on kernel
`6.14.0-37-generic` with NVIDIA `580.65.06`, but the newer installed kernel
`7.0.0-31-generic` is incomplete and is likely to be selected first by GRUB.

The Windows installation and shared EFI boot chain are present. Chromium's
selective site blocking is browser-local and is most likely caused by the enabled
BlockSite extension.

## Research state

- Both experiment queues are empty.
- `bqrunner` and `bqrunner2` are stopped.
- The hourly and mathematical review timers and services are stopped.
- No compute process is using the GPU.
- Results are summarized in `PAUSE_CHECKPOINT_2026-10-21.md`.

These services were stopped, not disabled. They can be resumed later.

## NVIDIA and Ubuntu boot evidence

Working current state:

- Running kernel: `6.14.0-37-generic`
- NVIDIA driver/module: `580.65.06`
- DKMS status for 6.14: installed
- `nvidia-smi`: succeeds; RTX 5080 visible; desktop rendering normally
- Secure Boot: disabled

Unsafe next-kernel state:

- `linux-image-7.0.0-31-generic` and its headers are half-configured.
- `/boot/initrd.img-7.0.0-31-generic` does not exist; the generic initrd symlink is
  therefore broken.
- NVIDIA `580.65.06` repeatedly failed to compile for kernel 7.0 because of kernel
  API incompatibilities recorded in the DKMS log.
- GRUB uses `GRUB_DEFAULT=0`, a hidden menu, and zero timeout. This combination is
  unsafe while the newest kernel is incomplete.
- NVIDIA `580.178.04` is available from the already-configured NVIDIA repository;
  the package-manager simulation upgrades the complete 580 package family.

Administrator repair sequence to run interactively:

```bash
sudo apt update
sudo apt install nvidia-driver-580-open
sudo apt --fix-broken install
```

Do not reboot if any command reports an error. If those commands succeed, verify:

```bash
dkms status
ls -lh /boot/initrd.img-7.0.0-31-generic
find /lib/modules/7.0.0-31-generic -path '*nvidia*.ko*' -print
sudo update-grub
```

Expected success conditions are an NVIDIA module for kernel 7.0, a real (not
missing) 7.0 initramfs, no half-configured packages, and a successful GRUB update.
Only then reboot Ubuntu. Adding Ubuntu's precompiled module package is a sensible
fallback if DKMS still fails, but should be simulated against the resulting package
state rather than installed blindly.

## Windows dual boot evidence

- The 949.3 GB NTFS Windows partition and 915 MB NTFS recovery partition exist.
- The 100 MB FAT32 EFI system partition is mounted and healthy enough to read.
- `EFI/Microsoft/Boot/bootmgfw.efi` and the Windows BCD files exist.
- Two active firmware entries point to Windows Boot Manager on that EFI partition.
- Firmware boot order is Ubuntu first, then Windows entries.

This verifies the Linux-visible boot chain, not the Windows OS or Windows NVIDIA
driver. Reaching the Windows desktop can only be proved by booting it. If Windows
must be used before Ubuntu is repaired, select **Windows Boot Manager** explicitly
from the firmware's one-time boot menu rather than allowing the default Ubuntu
entry to run.

## Chromium site blocking

System networking is not blocking the reported sites:

- DNS resolves Gmail, Forkable, and Facebook normally.
- Direct HTTPS requests reach all three with valid TLS and HTTP 200 responses
  (after Gmail's normal sign-in redirects).
- System proxy mode is `none`.
- `/etc/hosts` has no blocking entries.
- The only managed Chromium policy found disables incognito mode; it does not
  block URLs.

The Chromium profile has an enabled extension:

`BlockSite Block Websites & Stay Focused` (`eiimnmioipafcokbfikbljfdeojpcgbh`)

It was user-installed, has active access to `<all_urls>`, and is specifically made
to implement selective website blocking. Open `chrome://extensions`, disable
BlockSite temporarily, and reload Gmail/Forkable. If they work, either correct its
block list/schedule or remove it. Dark Reader is the only other non-built-in
extension found and is not a plausible selective blocker.

